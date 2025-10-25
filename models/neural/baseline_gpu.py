"""
GPU-Accelerated Neural Network Model for Work Operations Prediction
Uses the full HackathonDataset with all features including context rooms
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dataset.hackathon import HackathonDataset
from metrics.score import normalized_rooms_score

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ============================================================================
# 1. Custom Collate Function
# ============================================================================

def collate_fn(batch):
    """
    Custom collate function to handle variable-length context rooms
    """
    # Stack fixed-size tensors
    ids = torch.tensor([item['id'] for item in batch], dtype=torch.long)
    X = torch.stack([item['X'].float() for item in batch])  # visible operations (388,)
    Y = torch.stack([item['Y'].float() for item in batch])  # hidden operations (388,)
    room_cluster_one_hot = torch.stack([item['room_cluster_one_hot'].float() for item in batch])  # (11,)

    # Handle insurance company one-hot (shape can be (1, 14) or (14,))
    insurance_one_hots = []
    for item in batch:
        ins = item['insurance_company_one_hot'].float()
        if ins.ndim == 2:
            ins = ins.squeeze(0)  # (1, 14) -> (14,)
        insurance_one_hots.append(ins)
    insurance_company_one_hot = torch.stack(insurance_one_hots)  # (batch, 14)

    # Metadata features
    office_distance = torch.tensor([item['office_distance'] for item in batch], dtype=torch.float32)
    case_creation_year = torch.tensor([item['case_creation_year'] for item in batch], dtype=torch.float32)
    case_creation_month = torch.tensor([item['case_creation_month'] for item in batch], dtype=torch.float32)

    # Context rooms (variable length) - we'll encode them
    max_context_rooms = max(len(item['calculus']) for item in batch)

    if max_context_rooms > 0:
        # Create padded context tensor
        context_ops = torch.zeros(len(batch), max_context_rooms, 388, dtype=torch.float32)
        context_room_types = torch.zeros(len(batch), max_context_rooms, 11, dtype=torch.float32)
        context_mask = torch.zeros(len(batch), max_context_rooms, dtype=torch.bool)

        for i, item in enumerate(batch):
            for j, ctx in enumerate(item['calculus']):
                context_ops[i, j] = ctx['work_operations_index_encoded'].float()
                context_room_types[i, j] = ctx['room_cluster_one_hot'].float()
                context_mask[i, j] = True
    else:
        context_ops = torch.zeros(len(batch), 0, 388, dtype=torch.float32)
        context_room_types = torch.zeros(len(batch), 0, 11, dtype=torch.float32)
        context_mask = torch.zeros(len(batch), 0, dtype=torch.bool)

    return {
        'id': ids,
        'X': X,
        'Y': Y,
        'room_cluster_one_hot': room_cluster_one_hot,
        'insurance_company_one_hot': insurance_company_one_hot,
        'office_distance': office_distance,
        'case_creation_year': case_creation_year,
        'case_creation_month': case_creation_month,
        'context_ops': context_ops,
        'context_room_types': context_room_types,
        'context_mask': context_mask,
    }


# ============================================================================
# 2. Neural Network Architecture with Context Attention
# ============================================================================

class ContextAggregator(nn.Module):
    """Aggregates context room information using attention mechanism"""

    def __init__(self, op_dim=388, room_dim=11, hidden_dim=128):
        super(ContextAggregator, self).__init__()

        # Encode each context room
        self.context_encoder = nn.Sequential(
            nn.Linear(op_dim + room_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, context_ops, context_room_types, context_mask):
        """
        Args:
            context_ops: (batch, n_context, 388)
            context_room_types: (batch, n_context, 11)
            context_mask: (batch, n_context) - True where context exists
        Returns:
            aggregated_context: (batch, hidden_dim)
        """
        batch_size, n_context, _ = context_ops.shape

        if n_context == 0:
            # No context, return zeros
            return torch.zeros(batch_size, 128, device=context_ops.device)

        # Concatenate operations and room types
        context_features = torch.cat([context_ops, context_room_types], dim=-1)  # (batch, n_context, 399)

        # Encode each context room
        context_encoded = self.context_encoder(context_features)  # (batch, n_context, hidden_dim)

        # Compute attention weights
        attention_scores = self.attention(context_encoded).squeeze(-1)  # (batch, n_context)

        # Mask out padding
        attention_scores = attention_scores.masked_fill(~context_mask, float('-inf'))
        attention_weights = torch.softmax(attention_scores, dim=-1)  # (batch, n_context)

        # Handle case where all context is masked (shouldn't happen, but for safety)
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)

        # Weighted sum
        aggregated = torch.bmm(attention_weights.unsqueeze(1), context_encoded).squeeze(1)  # (batch, hidden_dim)

        return aggregated


class WorkOperationPredictor(nn.Module):
    """Neural network for predicting hidden work operations with context"""

    def __init__(self, use_context=True, hidden_dims=[512, 256, 128], output_dim=388, dropout=0.3):
        """
        Args:
            use_context: whether to use context rooms
            hidden_dims: list of hidden layer dimensions
            output_dim: number of operations to predict (388)
            dropout: dropout rate
        """
        super(WorkOperationPredictor, self).__init__()

        self.use_context = use_context

        # Context aggregator
        if use_context:
            self.context_aggregator = ContextAggregator(op_dim=388, room_dim=11, hidden_dim=128)
            context_dim = 128
        else:
            context_dim = 0

        # Input features:
        # - Visible operations: 388
        # - Room cluster one-hot: 11
        # - Insurance company one-hot: 14
        # - Metadata (year, month, distance): 3
        # - Context aggregation: 128 (if use_context)
        input_dim = 388 + 11 + 14 + 3 + context_dim

        # Build main network
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, X, room_cluster_one_hot, insurance_company_one_hot,
                year, month, distance, context_ops=None, context_room_types=None, context_mask=None):
        """
        Args:
            X: visible operations (batch, 388)
            room_cluster_one_hot: (batch, 11)
            insurance_company_one_hot: (batch, 14)
            year: (batch,)
            month: (batch,)
            distance: (batch,)
            context_ops: (batch, n_context, 388) - optional
            context_room_types: (batch, n_context, 11) - optional
            context_mask: (batch, n_context) - optional
        Returns:
            output: (batch, 388)
        """
        batch_size = X.shape[0]

        # Normalize metadata
        year_norm = (year - 2016) / 9.0  # 2016-2025 -> [0, 1]
        month_norm = (month - 1) / 11.0  # 1-12 -> [0, 1]
        distance_norm = torch.log1p(distance) / torch.log1p(torch.tensor(922.4))  # log normalization

        metadata = torch.stack([year_norm, month_norm, distance_norm], dim=1)  # (batch, 3)

        # Concatenate base features
        features = torch.cat([
            X,
            room_cluster_one_hot,
            insurance_company_one_hot,
            metadata,
        ], dim=1)

        # Add context if available
        if self.use_context and context_ops is not None:
            context_agg = self.context_aggregator(context_ops, context_room_types, context_mask)
            features = torch.cat([features, context_agg], dim=1)

        # Forward through network
        output = self.network(features)

        return output


# ============================================================================
# 3. Training Loop
# ============================================================================

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20, device=DEVICE):
    """Train the neural network"""

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            # Move batch to device
            X = batch['X'].to(device)
            Y = batch['Y'].to(device)
            room_cluster = batch['room_cluster_one_hot'].to(device)
            insurance = batch['insurance_company_one_hot'].to(device)
            year = batch['case_creation_year'].to(device)
            month = batch['case_creation_month'].to(device)
            distance = batch['office_distance'].to(device)
            context_ops = batch['context_ops'].to(device)
            context_room_types = batch['context_room_types'].to(device)
            context_mask = batch['context_mask'].to(device)

            # Forward pass
            optimizer.zero_grad()
            output = model(X, room_cluster, insurance, year, month, distance,
                          context_ops, context_room_types, context_mask)
            loss = criterion(output, Y)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                X = batch['X'].to(device)
                Y = batch['Y'].to(device)
                room_cluster = batch['room_cluster_one_hot'].to(device)
                insurance = batch['insurance_company_one_hot'].to(device)
                year = batch['case_creation_year'].to(device)
                month = batch['case_creation_month'].to(device)
                distance = batch['office_distance'].to(device)
                context_ops = batch['context_ops'].to(device)
                context_room_types = batch['context_room_types'].to(device)
                context_mask = batch['context_mask'].to(device)

                output = model(X, room_cluster, insurance, year, month, distance,
                              context_ops, context_room_types, context_mask)
                loss = criterion(output, Y)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            print(f"  → New best model (Val Loss={best_val_loss:.4f})")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


# ============================================================================
# 4. Prediction and Evaluation
# ============================================================================

def predict_operations(model, dataloader, threshold=0.5, device=DEVICE):
    """Generate predictions for all rooms"""

    model.eval()
    predictions = {}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating predictions"):
            X = batch['X'].to(device)
            room_cluster = batch['room_cluster_one_hot'].to(device)
            insurance = batch['insurance_company_one_hot'].to(device)
            year = batch['case_creation_year'].to(device)
            month = batch['case_creation_month'].to(device)
            distance = batch['office_distance'].to(device)
            context_ops = batch['context_ops'].to(device)
            context_room_types = batch['context_room_types'].to(device)
            context_mask = batch['context_mask'].to(device)
            ids = batch['id'].cpu().numpy()

            # Forward pass
            output = model(X, room_cluster, insurance, year, month, distance,
                          context_ops, context_room_types, context_mask)
            probs = torch.sigmoid(output)

            # Apply threshold
            pred_ops = (probs > threshold).cpu().numpy()

            # Store predictions
            for i, room_id in enumerate(ids):
                predicted_codes = np.where(pred_ops[i])[0].tolist()
                predictions[int(room_id)] = predicted_codes

    return predictions


def evaluate_predictions(predictions, val_dataset):
    """Evaluate predictions using the normalized rooms score"""

    y_true = []
    y_pred = []

    # Build ground truth from dataset
    for idx in range(len(val_dataset)):
        sample = val_dataset[idx]
        room_id = sample['id']

        if room_id not in predictions:
            continue

        # Ground truth
        true_vec = sample['Y'].numpy().astype(int).tolist()

        # Prediction
        pred_vec = [0] * 388
        for op in predictions[room_id]:
            pred_vec[op] = 1

        y_true.append(true_vec)
        y_pred.append(pred_vec)

    score = normalized_rooms_score(y_true, y_pred)
    return score


# ============================================================================
# 5. Main Execution
# ============================================================================

def main():
    print("="*80)
    print("GPU-ACCELERATED NEURAL NETWORK MODEL WITH CONTEXT")
    print("="*80)

    # Load datasets
    print("\n--- Loading datasets ---")
    train_dataset = HackathonDataset(split="train", download=False, seed=42, root="data", fraction=0.4)
    val_dataset = HackathonDataset(split="val", download=False, seed=42, root="data")
    test_dataset = HackathonDataset(split="test", download=False, seed=42, root="data")

    print(f"Train dataset: {len(train_dataset)} rooms")
    print(f"Val dataset: {len(val_dataset)} rooms")
    print(f"Test dataset: {len(test_dataset)} rooms")

    # Create dataloaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           collate_fn=collate_fn, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=4)

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    # Initialize model
    print("\n--- Initializing model ---")
    model = WorkOperationPredictor(
        use_context=True,
        hidden_dims=[512, 256, 128],
        output_dim=388,
        dropout=0.3
    ).to(DEVICE)

    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # Train model
    print("\n--- Training model ---")
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=20,
        device=DEVICE
    )

    # Evaluate on validation set
    print("\n--- Evaluating on validation set ---")
    val_predictions = predict_operations(model, val_loader, threshold=0.5, device=DEVICE)

    print("\n--- Calculating validation score ---")
    val_score = evaluate_predictions(val_predictions, val_dataset)
    print(f"Validation Score: {val_score:.4f}")

    # Statistics
    num_empty = sum(1 for preds in val_predictions.values() if len(preds) == 0)
    avg_predictions = np.mean([len(preds) for preds in val_predictions.values()])
    print(f"\nPrediction Statistics:")
    print(f"  Empty predictions: {num_empty}/{len(val_predictions)} ({100*num_empty/len(val_predictions):.2f}%)")
    print(f"  Avg predictions per room: {avg_predictions:.2f}")

    # Generate test predictions
    print("\n--- Generating test predictions ---")
    test_predictions = predict_operations(model, test_loader, threshold=0.5, device=DEVICE)

    # Create submission
    print("\n--- Creating submission ---")
    os.makedirs("submissions", exist_ok=True)
    test_dataset.create_submission(test_predictions)

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Final validation score: {val_score:.4f}")
    print(f"Submission file created in submissions/")


if __name__ == "__main__":
    main()
