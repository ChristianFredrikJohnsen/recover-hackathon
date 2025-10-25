"""
Correlation-Based Baseline with Hyperparameter Tuning

This model uses association rules and statistical correlation with grid search
to find optimal thresholds.

Approach:
1. Learn high-confidence association rules from training data
2. Grid search over hyperparameters on validation set
3. Use best parameters for final predictions
"""

import os
import sys
import numpy as np
import polars as pl
from collections import defaultdict
from itertools import product

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dataset.hackathon import HackathonDataset
from metrics import normalized_rooms_score

print("="*80)
print("CORRELATION-BASED BASELINE WITH HYPERPARAMETER TUNING")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/5] Loading datasets...")

train_dataset = HackathonDataset(split="train", download=False, seed=42, root="data", fraction=0.1)
val_dataset = HackathonDataset(split="val", download=False, seed=42, root="data")
test_dataset = HackathonDataset(split="test", download=False, seed=42, root="data")

print(f"Train size: {len(train_dataset)}")
print(f"Val size: {len(val_dataset)}")
print(f"Test size: {len(test_dataset)}")

# ============================================================================
# 2. BUILD ASSOCIATION RULES
# ============================================================================
print("\n[2/5] Building association rules from training data...")

train_df = train_dataset.get_polars_dataframe()

train_visible = train_df.filter(pl.col("is_hidden") == False)
train_hidden = train_df.filter(pl.col("is_hidden") == True)

room_visible = train_visible.group_by(["project_id", "room"]).agg([
    pl.col("work_operation").alias("visible_ops"),
    pl.col("room_cluster").first(),
])

room_hidden = train_hidden.group_by(["project_id", "room"]).agg([
    pl.col("work_operation").alias("hidden_ops"),
])

room_data = room_visible.join(room_hidden, on=["project_id", "room"], how="left")

# Build association rules: IF operation A is visible THEN predict operation B
# Calculate: support, confidence, lift

rules = {}  # (vis_op, hid_op) -> {'confidence': ..., 'lift': ..., 'support': ...}
operation_counts = defaultdict(int)  # Total occurrences of each hidden operation
visible_counts = defaultdict(int)    # Total occurrences of each visible operation
total_rooms = len(room_data)
rooms_with_hidden = 0

print("Calculating association rules...")

for row in room_data.iter_rows(named=True):
    visible = set(row['visible_ops'] or [])
    hidden = set(row['hidden_ops'] or [])

    if hidden:
        rooms_with_hidden += 1

    # Count individual operations
    for v_op in visible:
        visible_counts[v_op] += 1

    for h_op in hidden:
        operation_counts[h_op] += 1

    # Count co-occurrences
    for v_op in visible:
        for h_op in hidden:
            if (v_op, h_op) not in rules:
                rules[(v_op, h_op)] = {'count': 0}
            rules[(v_op, h_op)]['count'] += 1

# Calculate metrics for each rule
print("Computing rule metrics (confidence, lift, support)...")

for (v_op, h_op), stats in rules.items():
    count = stats['count']

    # Confidence: P(hidden | visible)
    confidence = count / visible_counts[v_op] if visible_counts[v_op] > 0 else 0

    # Support: P(visible AND hidden)
    support = count / total_rooms

    # Lift: P(hidden | visible) / P(hidden)
    p_hidden = operation_counts[h_op] / total_rooms if total_rooms > 0 else 0
    lift = confidence / p_hidden if p_hidden > 0 else 0

    stats['confidence'] = confidence
    stats['support'] = support
    stats['lift'] = lift

print(f"Total rules generated: {len(rules)}")
print(f"Total rooms: {total_rooms}")
print(f"Rooms with hidden operations: {rooms_with_hidden} ({rooms_with_hidden/total_rooms:.1%})")

# Show top rules by confidence
print("\nTop 10 rules by confidence:")
code_to_wo = train_dataset.work_operations_dataset.code_to_wo
top_rules = sorted(rules.items(), key=lambda x: (x[1]['confidence'], x[1]['count']), reverse=True)[:10]

for (v_op, h_op), stats in top_rules:
    v_name = code_to_wo.get(v_op, f"Op{v_op}")[:35]
    h_name = code_to_wo.get(h_op, f"Op{h_op}")[:35]
    print(f"  [{v_op:3d}→{h_op:3d}] Conf={stats['confidence']:.3f}, "
          f"Lift={stats['lift']:.2f}, Supp={stats['support']:.4f}, Count={stats['count']}")
    print(f"    IF '{v_name}' THEN '{h_name}'")

# ============================================================================
# 3. ROOM-SPECIFIC STATISTICS
# ============================================================================
print("\n[3/5] Computing room-specific statistics...")

room_stats = defaultdict(lambda: {'total': 0, 'with_hidden': 0})

for row in room_data.iter_rows(named=True):
    room_cluster = row['room_cluster']
    hidden = row['hidden_ops'] or []

    room_stats[room_cluster]['total'] += 1
    if hidden:
        room_stats[room_cluster]['with_hidden'] += 1

# Calculate empty rate per room type
room_empty_rate = {}
for room_type, stats in room_stats.items():
    room_empty_rate[room_type] = 1 - (stats['with_hidden'] / stats['total'])

print("Empty rate by room type:")
for room_type, rate in sorted(room_empty_rate.items(), key=lambda x: x[1], reverse=True):
    print(f"  {room_type:20s}: {rate:6.1%}")

# ============================================================================
# 4. PREDICTION FUNCTION
# ============================================================================
print("\n[4/5] Creating prediction function...")

def predict_operations(visible_ops, room_cluster, rules,
                       min_confidence=0.3, min_lift=1.5, min_support=0.001,
                       empty_threshold=0.7, max_predictions=10):
    """
    Predict hidden operations using association rules.

    Args:
        visible_ops: List of visible operation codes
        room_cluster: Room type
        rules: Dictionary of association rules
        min_confidence: Minimum rule confidence
        min_lift: Minimum rule lift
        min_support: Minimum rule support
        empty_threshold: Room types above this empty rate get no predictions
        max_predictions: Maximum number of predictions

    Returns:
        List of predicted operation codes
    """

    # If room type is mostly empty, predict nothing
    if room_empty_rate.get(room_cluster, 0) >= empty_threshold:
        return []

    # Collect candidate predictions with scores
    candidates = {}  # op_code -> score

    visible_set = set(visible_ops)

    for v_op in visible_ops:
        # Find all rules where this operation is the antecedent
        for (vis, hid), stats in rules.items():
            if vis == v_op and hid not in visible_set:
                # Check if rule meets thresholds
                if (stats['confidence'] >= min_confidence and
                    stats['lift'] >= min_lift and
                    stats['support'] >= min_support):

                    # Score: weighted combination of confidence and lift
                    score = stats['confidence'] * np.sqrt(stats['lift'])

                    # Track best score for each operation
                    if hid not in candidates or score > candidates[hid]:
                        candidates[hid] = score

    # Sort by score and return top predictions
    sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    predictions = [op for op, _ in sorted_candidates[:max_predictions]]

    return predictions

# ============================================================================
# 5. HYPERPARAMETER TUNING
# ============================================================================
print("\n[5/5] Tuning hyperparameters on validation set...")

# Load validation data
val_df = val_dataset.get_polars_dataframe()
val_visible = val_df.filter(pl.col("is_hidden") == False)
val_hidden = val_df.filter(pl.col("is_hidden") == True)

val_room_visible = val_visible.group_by(["project_id", "room"]).agg([
    pl.col("work_operation").alias("visible_ops"),
    pl.col("room_cluster").first(),
])

val_room_hidden = val_hidden.group_by(["project_id", "room"]).agg([
    pl.col("work_operation").alias("hidden_ops"),
])

val_room_data = val_room_visible.join(val_room_hidden, on=["project_id", "room"], how="left")

# Prepare validation data
val_visible_list = []
val_hidden_list = []
val_room_types = []

for row in val_room_data.iter_rows(named=True):
    val_visible_list.append(row['visible_ops'] or [])
    val_hidden_list.append(row['hidden_ops'] or [])
    val_room_types.append(row['room_cluster'])

print(f"Validation set: {len(val_visible_list)} rooms")

# Grid search over hyperparameters
param_grid = {
    'min_confidence': [0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    'min_lift': [1.2, 1.5, 2.0, 2.5],
    'min_support': [0.0001, 0.0005, 0.001],
    'empty_threshold': [0.65, 0.70, 0.75],
    'max_predictions': [3, 5, 7, 10]
}

print("\nGrid search parameters:")
for param, values in param_grid.items():
    print(f"  {param}: {values}")

total_combinations = np.prod([len(v) for v in param_grid.values()])
print(f"\nTotal combinations to test: {total_combinations}")

best_score = -float('inf')
best_params = None
best_predictions = None

# Grid search
print("\nRunning grid search...")
print("Format: [config_num/total] score | params")

config_num = 0

for min_conf, min_lift, min_supp, empty_thresh, max_pred in product(
    param_grid['min_confidence'],
    param_grid['min_lift'],
    param_grid['min_support'],
    param_grid['empty_threshold'],
    param_grid['max_predictions']
):
    config_num += 1

    # Generate predictions for this configuration
    preds = []
    for visible, room_type in zip(val_visible_list, val_room_types):
        pred = predict_operations(
            visible, room_type, rules,
            min_confidence=min_conf,
            min_lift=min_lift,
            min_support=min_supp,
            empty_threshold=empty_thresh,
            max_predictions=max_pred
        )
        preds.append(pred)

    # Evaluate
    score = normalized_rooms_score(preds, val_hidden_list)

    # Track best
    if score > best_score:
        best_score = score
        best_params = {
            'min_confidence': min_conf,
            'min_lift': min_lift,
            'min_support': min_supp,
            'empty_threshold': empty_thresh,
            'max_predictions': max_pred
        }
        best_predictions = preds

        print(f"✓ [{config_num:4d}/{total_combinations}] {score:.4f} | conf={min_conf:.2f}, lift={min_lift:.1f}, "
              f"supp={min_supp:.4f}, empty={empty_thresh:.2f}, max={max_pred}")

    # Show progress every 50 configs
    if config_num % 50 == 0:
        print(f"  [{config_num:4d}/{total_combinations}] Progress... (best so far: {best_score:.4f})")

print("\n" + "="*80)
print("GRID SEARCH COMPLETE")
print("="*80)

print(f"\nBest validation score: {best_score:.4f}")
print("\nBest hyperparameters:")
for param, value in best_params.items():
    print(f"  {param:20s}: {value}")

# Statistics with best params
avg_preds = np.mean([len(p) for p in best_predictions])
empty_preds = sum(1 for p in best_predictions if len(p) == 0)

print(f"\nPrediction statistics (best config):")
print(f"  Average predictions per room: {avg_preds:.2f}")
print(f"  Empty predictions: {empty_preds}/{len(best_predictions)} ({empty_preds/len(best_predictions):.1%})")

# Show some example predictions
print("\nSample predictions (best config):")
for i in range(min(5, len(best_predictions))):
    if len(best_predictions[i]) > 0 or len(val_hidden_list[i]) > 0:
        correct = len(set(best_predictions[i]) & set(val_hidden_list[i]))
        print(f"\nRoom {i} ({val_room_types[i]}):")
        print(f"  Predicted: {best_predictions[i][:5]}")
        print(f"  Actual:    {val_hidden_list[i][:5]}")
        print(f"  Correct:   {correct}/{len(val_hidden_list[i])}")

# ============================================================================
# 6. GENERATE TEST PREDICTIONS WITH BEST PARAMS
# ============================================================================
print("\n[6/6] Generating test predictions with best hyperparameters...")

test_df = test_dataset.get_polars_dataframe()

test_room_data = test_df.group_by(["project_id", "room"]).agg([
    pl.col("work_operation").alias("visible_ops"),
    pl.col("room_cluster").first(),
    pl.col("index").first().alias("room_id"),
])

test_predictions = {}

for row in test_room_data.iter_rows(named=True):
    visible = row['visible_ops'] or []
    room_cluster = row['room_cluster']
    room_id = row['room_id']

    pred = predict_operations(
        visible, room_cluster, rules,
        **best_params  # Use best hyperparameters
    )
    test_predictions[room_id] = pred

# Create submission
test_dataset.create_submission(test_predictions)

print("\n" + "="*80)
print("TUNED CORRELATION BASELINE COMPLETE!")
print("="*80)
print(f"\nFinal validation score: {best_score:.4f}")
print(f"Test predictions generated: {len(test_predictions)} rooms")
print("\nSubmission file saved to submissions/ folder")
