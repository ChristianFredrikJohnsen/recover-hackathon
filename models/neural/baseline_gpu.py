"""
GPU-Accelerated Neural Network Model for Work Operations Prediction
Uses pandas dataframe representation and PyTorch for training
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dataset.hackathon import HackathonDataset
from metrics.score import normalized_rooms_score

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ============================================================================
# 1. Data Preprocessing
# ============================================================================

class OperationDataset(Dataset):
    """PyTorch dataset for room operations"""

    def __init__(self, df, num_operations=388, is_test=False):
        """
        Args:
            df: pandas dataframe from get_pandas_dataframe()
            num_operations: total number of work operations (388)
            is_test: if True, all operations are visible (no hidden)
        """
        self.num_operations = num_operations
        self.is_test = is_test

        # Group by room (index)
        self.room_indices = df['index'].unique()

        # Create room -> operations mapping
        self.room_data = {}
        for room_idx in self.room_indices:
            room_df = df[df['index'] == room_idx]

            # Get visible operations
            visible_ops = room_df[~room_df['is_hidden']]['work_operation'].values

            # Get hidden operations (target)
            hidden_ops = room_df[room_df['is_hidden']]['work_operation'].values

            # Get metadata (take first row since it's the same for all operations in a room)
            row = room_df.iloc[0]

            # Parse insurance company one-hot (it's stored as string representation of list)
            insurance_one_hot = eval(row['insurance_company_one_hot'])

            self.room_data[room_idx] = {
                'visible_ops': visible_ops,
                'hidden_ops': hidden_ops,
                'room_cluster': row['room_cluster'],
                'insurance_one_hot': insurance_one_hot,
                'year': row['case_creation_year'],
                'month': int(row['case_creation_month']),
                'office_distance': row['office_distance'],
                'project_id': row['project_id'],
            }

        # Encode room clusters
        self.room_clusters = sorted(df['room_cluster'].unique())
        self.room_cluster_to_idx = {cluster: idx for idx, cluster in enumerate(self.room_clusters)}

        print(f"Loaded {len(self.room_indices)} rooms")
        print(f"Room clusters: {self.room_clusters}")

    def __len__(self):
        return len(self.room_indices)

    def __getitem__(self, idx):
        room_idx = self.room_indices[idx]
        data = self.room_data[room_idx]

        # Create visible operations one-hot vector
        visible_vec = np.zeros(self.num_operations, dtype=np.float32)
        visible_vec[data['visible_ops']] = 1.0

        # Create target (hidden operations) one-hot vector
        target_vec = np.zeros(self.num_operations, dtype=np.float32)
        if not self.is_test:
            target_vec[data['hidden_ops']] = 1.0

        # Create room cluster one-hot
        room_cluster_vec = np.zeros(len(self.room_clusters), dtype=np.float32)
        room_cluster_vec[self.room_cluster_to_idx[data['room_cluster']]] = 1.0

        # Insurance company one-hot
        insurance_vec = np.array(data['insurance_one_hot'], dtype=np.float32)

        # Normalize year to [0, 1] range (2016-2025)
        year_normalized = (data['year'] - 2016) / 9.0

        # Normalize month to [0, 1] range
        month_normalized = (data['month'] - 1) / 11.0

        # Normalize office distance (log scale + normalization)
        distance_normalized = np.log1p(data['office_distance']) / np.log1p(922.4)  # max distance

        # Concatenate all features
        features = np.concatenate([
            visible_vec,  # 388 dims
            room_cluster_vec,  # 12 dims
            insurance_vec,  # 14 dims
            [year_normalized, month_normalized, distance_normalized]  # 3 dims
        ])

        return {
            'room_idx': room_idx,
            'features': torch.from_numpy(features),
            'target': torch.from_numpy(target_vec),
            'project_id': data['project_id'],
        }


# ============================================================================
# 2. Neural Network Architecture
# ============================================================================

class WorkOperationPredictor(nn.Module):
    """Neural network for predicting hidden work operations"""

    def __init__(self, input_dim=417, hidden_dims=[512, 256, 128], output_dim=388, dropout=0.3):
        """
        Args:
            input_dim: total input features (388 ops + 12 room + 14 insurance + 3 metadata)
            hidden_dims: list of hidden layer dimensions
            output_dim: number of operations to predict (388)
            dropout: dropout rate
        """
        super(WorkOperationPredictor, self).__init__()

        layers = []
        prev_dim = input_dim

        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


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
            features = batch['features'].to(device)
            target = batch['target'].to(device)

            # Forward pass
            optimizer.zero_grad()
            output = model(features)
            loss = criterion(output, target)

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
                features = batch['features'].to(device)
                target = batch['target'].to(device)

                output = model(features)
                loss = criterion(output, target)
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
            features = batch['features'].to(device)
            room_indices = batch['room_idx'].numpy()

            # Forward pass
            output = model(features)
            probs = torch.sigmoid(output)

            # Apply threshold to get binary predictions
            pred_ops = (probs > threshold).cpu().numpy()

            # Store predictions
            for i, room_idx in enumerate(room_indices):
                predicted_codes = np.where(pred_ops[i])[0].tolist()
                predictions[int(room_idx)] = predicted_codes

    return predictions


def evaluate_predictions(predictions, val_dataset_hackathon):
    """Evaluate predictions using the normalized rooms score"""

    # Get ground truth from validation dataset
    val_df = val_dataset_hackathon.get_pandas_dataframe()

    y_true = []
    y_pred = []

    for room_idx in sorted(predictions.keys()):
        # Get ground truth hidden operations for this room
        room_df = val_df[val_df['index'] == room_idx]
        true_hidden = room_df[room_df['is_hidden']]['work_operation'].values.tolist()

        # Get visible operations (needed for scoring function)
        visible = room_df[~room_df['is_hidden']]['work_operation'].values.tolist()

        # Create one-hot vectors
        true_vec = [0] * 388
        for op in true_hidden:
            true_vec[op] = 1

        pred_vec = [0] * 388
        for op in predictions[room_idx]:
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
    print("GPU-ACCELERATED NEURAL NETWORK MODEL")
    print("="*80)

    # Load datasets
    print("\n--- Loading datasets ---")
    train_dataset_hackathon = HackathonDataset(split="train", download=False, seed=42, root="data", fraction=0.3)
    val_dataset_hackathon = HackathonDataset(split="val", download=False, seed=42, root="data")
    test_dataset_hackathon = HackathonDataset(split="test", download=False, seed=42, root="data")

    # Convert to pandas
    print("\n--- Converting to pandas ---")
    train_df = train_dataset_hackathon.get_pandas_dataframe()
    val_df = val_dataset_hackathon.get_pandas_dataframe()
    test_df = test_dataset_hackathon.get_pandas_dataframe()

    print(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")

    # Create PyTorch datasets
    print("\n--- Creating PyTorch datasets ---")
    train_dataset = OperationDataset(train_df, num_operations=388, is_test=False)
    val_dataset = OperationDataset(val_df, num_operations=388, is_test=False)
    test_dataset = OperationDataset(test_df, num_operations=388, is_test=True)

    # Create dataloaders
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    # Initialize model
    print("\n--- Initializing model ---")
    input_dim = 388 + 12 + 14 + 3  # operations + room clusters + insurance + metadata
    model = WorkOperationPredictor(
        input_dim=input_dim,
        hidden_dims=[512, 256, 128],
        output_dim=388,
        dropout=0.3
    ).to(DEVICE)

    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy with logits
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

    # Calculate custom score
    print("\n--- Calculating validation score ---")
    val_score = evaluate_predictions(val_predictions, val_dataset_hackathon)
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
    test_dataset_hackathon.create_submission(test_predictions)

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Final validation score: {val_score:.4f}")
    print(f"Submission file created in submissions/")


if __name__ == "__main__":
    main()
