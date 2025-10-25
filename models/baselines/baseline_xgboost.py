"""
Minimal XGBoost Baseline for Recover Hackathon

This script creates a simple baseline model using XGBoost to predict missing work operations.
The approach:
1. Load data as tabular format
2. Create simple features from visible operations + room type + metadata
3. Train one binary classifier per operation (multi-label classification)
4. Generate predictions for test set
"""

import os
import sys
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from tqdm import tqdm
import xgboost as xgb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dataset.hackathon import HackathonDataset
from metrics import normalized_rooms_score

print("="*80)
print("XGBOOST BASELINE MODEL")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/5] Loading datasets...")

# Load small fraction for speed
train_dataset = HackathonDataset(split="train", download=False, seed=42, root="data", fraction=0.1)
val_dataset = HackathonDataset(split="val", download=False, seed=42, root="data")
test_dataset = HackathonDataset(split="test", download=False, seed=42, root="data")

print(f"Train size: {len(train_dataset)}")
print(f"Val size: {len(val_dataset)}")
print(f"Test size: {len(test_dataset)}")

# Get as dataframes
train_df = train_dataset.get_polars_dataframe()
val_df = val_dataset.get_polars_dataframe()
test_df = test_dataset.get_polars_dataframe()

print(f"Train dataframe shape: {train_df.shape}")
print(f"Val dataframe shape: {val_df.shape}")
print(f"Test dataframe shape: {test_df.shape}")

# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================
print("\n[2/5] Creating features...")

def create_features(df, is_train=True):
    """
    Create simple features from the dataframe.

    Features:
    - One-hot encoded work operations (only visible ones if training)
    - Room cluster (one-hot)
    - Insurance company (one-hot)
    - Office distance
    - Case creation year/month
    - Number of operations in the room
    """

    # Filter out hidden operations for training/val
    if is_train:
        df_visible = df.filter(pl.col("is_hidden") == False)
    else:
        df_visible = df

    # Group by room (project_id + room combination)
    room_groups = df_visible.group_by(["project_id", "room"]).agg([
        pl.col("work_operation").alias("operations"),
        pl.col("room_cluster").first(),
        pl.col("insurance_company").first(),
        pl.col("office_distance").first(),
        pl.col("case_creation_year").first(),
        pl.col("case_creation_month").first(),
    ])

    # Create operation features (multi-hot encoding)
    num_operations = 388

    def operations_to_multihot(ops_list):
        vec = [0] * num_operations
        for op in ops_list:
            if 0 <= op < num_operations:
                vec[op] = 1
        return vec

    # Convert to pandas for easier manipulation
    room_df = room_groups.to_pandas()

    # Multi-hot encode operations
    operation_features = np.array([operations_to_multihot(ops) for ops in room_df['operations']])

    # One-hot encode room cluster
    room_cluster_dummies = pd.get_dummies(room_df['room_cluster'], prefix='room')

    # One-hot encode insurance company
    insurance_dummies = pd.get_dummies(room_df['insurance_company'], prefix='ins')

    # Create feature matrix
    X = pd.DataFrame(operation_features, columns=[f'op_{i}' for i in range(num_operations)])
    X = pd.concat([X, room_cluster_dummies, insurance_dummies], axis=1)
    X['office_distance'] = room_df['office_distance'].values
    X['year'] = room_df['case_creation_year'].values
    X['month'] = room_df['case_creation_month'].astype(str).values
    X['num_ops'] = room_df['operations'].apply(len).values

    # One-hot encode month
    month_dummies = pd.get_dummies(X['month'], prefix='month')
    X = pd.concat([X.drop('month', axis=1), month_dummies], axis=1)

    # Store identifiers
    X['project_id'] = room_df['project_id'].values
    X['room'] = room_df['room'].values

    return X, room_df

print("Creating train features...")
X_train, train_rooms = create_features(train_df, is_train=True)

print("Creating val features...")
X_val, val_rooms = create_features(val_df, is_train=True)

print("Creating test features...")
X_test, test_rooms = create_features(test_df, is_train=False)

print(f"Feature matrix shape: {X_train.shape}")
print(f"Features: {list(X_train.columns[:10])}... (showing first 10)")

# ============================================================================
# 3. CREATE LABELS
# ============================================================================
print("\n[3/5] Creating labels...")

def create_labels(df):
    """Create multi-label targets (hidden operations per room)"""

    df_hidden = df.filter(pl.col("is_hidden") == True)

    room_labels = df_hidden.group_by(["project_id", "room"]).agg([
        pl.col("work_operation").alias("hidden_ops")
    ])

    room_labels_df = room_labels.to_pandas()

    num_operations = 388

    def operations_to_multihot(ops_list):
        vec = [0] * num_operations
        for op in ops_list:
            if 0 <= op < num_operations:
                vec[op] = 1
        return vec

    y = np.array([operations_to_multihot(ops) for ops in room_labels_df['hidden_ops']])

    return y, room_labels_df

y_train, train_labels = create_labels(train_df)
y_val, val_labels = create_labels(val_df)

print(f"Train labels shape: {y_train.shape}")
print(f"Val labels shape: {y_val.shape}")

# ============================================================================
# 4. TRAIN MODEL
# ============================================================================
print("\n[4/5] Training XGBoost models...")

# Separate features from identifiers
feature_cols = [col for col in X_train.columns if col not in ['project_id', 'room']]
X_train_features = X_train[feature_cols]
X_val_features = X_val[feature_cols]
X_test_features = X_test[feature_cols]

print(f"Number of features: {len(feature_cols)}")

# We'll train classifiers for the top N most common operations only (for speed)
# This is a simplification - a full solution would train for all operations

# Find most common operations in training set
operation_counts = y_train.sum(axis=0)
top_k = 50  # Train only for top 50 operations
top_operations = np.argsort(operation_counts)[-top_k:][::-1]

print(f"Training models for top {top_k} operations")
print(f"These operations cover {operation_counts[top_operations].sum() / y_train.sum() * 100:.1f}% of all hidden operations")

models = {}

for op_idx in tqdm(top_operations, desc="Training models"):
    if operation_counts[op_idx] < 10:  # Skip very rare operations
        continue

    # Train binary classifier for this operation
    y_train_op = y_train[:, op_idx]

    # Skip if no positive examples
    if y_train_op.sum() == 0:
        continue

    # XGBoost parameters for binary classification
    params = {
        'objective': 'binary:logistic',
        'max_depth': 4,
        'learning_rate': 0.1,
        'n_estimators': 50,
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'random_state': 42,
        'n_jobs': -1,
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train_features, y_train_op, verbose=False)

    models[op_idx] = model

print(f"Trained {len(models)} models")

# ============================================================================
# 5. GENERATE PREDICTIONS
# ============================================================================
print("\n[5/5] Generating predictions...")

def predict(X_features, models, top_operations, threshold=0.5, top_n=10):
    """
    Generate predictions for all rooms.

    Returns:
        List of lists, where each inner list contains predicted operation codes for a room
    """
    predictions = []

    for idx in range(len(X_features)):
        room_preds = []
        scores = []

        # Get predictions from all models
        for op_idx, model in models.items():
            pred_proba = model.predict_proba(X_features.iloc[idx:idx+1])[0, 1]

            if pred_proba > threshold:
                scores.append((op_idx, pred_proba))

        # Sort by confidence and take top N
        scores.sort(key=lambda x: x[1], reverse=True)
        room_preds = [op for op, _ in scores[:top_n]]

        predictions.append(room_preds)

    return predictions

# Validation predictions
print("Predicting on validation set...")
val_preds = predict(X_val_features, models, top_operations, threshold=0.3, top_n=10)

# Convert val labels to list format
val_targets = []
for idx in range(len(y_val)):
    targets = [i for i, v in enumerate(y_val[idx]) if v == 1]
    val_targets.append(targets)

# Evaluate
val_score = normalized_rooms_score(val_preds, val_targets)
print(f"\nValidation Score: {val_score:.4f}")

# Test predictions
print("\nPredicting on test set...")
test_preds = predict(X_test_features, models, top_operations, threshold=0.3, top_n=10)

# ============================================================================
# 6. CREATE SUBMISSION
# ============================================================================
print("\n[6/6] Creating submission file...")

# Get test IDs
test_ids_df = test_df.select(["project_id", "room"]).unique().sort(["project_id", "room"])
test_ids = test_ids_df.to_pandas()

# Map predictions to IDs
submission_predictions = {}
for idx, row in test_ids.iterrows():
    # Find corresponding prediction
    mask = (X_test['project_id'] == row['project_id']) & (X_test['room'] == row['room'])
    pred_idx = X_test[mask].index[0]

    # Get the actual test ID from test_df
    test_room = test_df.filter(
        (pl.col("project_id") == row['project_id']) &
        (pl.col("room") == row['room'])
    )
    test_id = test_room.select("id").unique()[0, 0]

    submission_predictions[test_id] = test_preds[pred_idx]

# Create submission using dataset method
test_dataset.create_submission(submission_predictions)

print("\n" + "="*80)
print("BASELINE COMPLETE!")
print("="*80)
print(f"\nValidation Score: {val_score:.4f}")
print(f"Predicted for {len(submission_predictions)} test rooms")
print("\nSubmission file created in submissions/ folder")
print("\nNext steps to improve:")
print("1. Train models for all 388 operations (not just top 50)")
print("2. Use better features (co-occurrence patterns, context from other rooms)")
print("3. Tune hyperparameters")
print("4. Try different thresholds and top_n values")
print("5. Use ensemble methods")
