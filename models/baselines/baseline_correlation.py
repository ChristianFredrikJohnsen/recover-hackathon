"""
Correlation-Based Baseline with One-Hot Encoding

This model uses statistical correlation between operations to make predictions.

Approach:
1. Create co-occurrence matrix: how often operations appear together
2. Calculate correlation between visible and hidden operations
3. For each room, predict operations most correlated with visible ones
4. Use room type as additional signal
"""

import os
import sys
import numpy as np
import polars as pl
from collections import defaultdict
from scipy.stats import chi2_contingency

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dataset.hackathon import HackathonDataset
from metrics import normalized_rooms_score

print("="*80)
print("CORRELATION-BASED BASELINE")
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
# 2. BUILD CO-OCCURRENCE MATRIX
# ============================================================================
print("\n[2/5] Building co-occurrence matrix...")

train_df = train_dataset.get_polars_dataframe()
NUM_OPS = 388

# Create binary matrix for each room: which operations are present
train_visible = train_df.filter(pl.col("is_hidden") == False)
train_hidden = train_df.filter(pl.col("is_hidden") == True)

# Group by room
room_visible = train_visible.group_by(["project_id", "room"]).agg([
    pl.col("work_operation").alias("visible_ops"),
    pl.col("room_cluster").first(),
])

room_hidden = train_hidden.group_by(["project_id", "room"]).agg([
    pl.col("work_operation").alias("hidden_ops"),
])

room_data = room_visible.join(room_hidden, on=["project_id", "room"], how="left")

# Build matrices for correlation analysis
# For each room, create binary vectors of which operations are present
print("Creating binary matrices...")

visible_matrix = np.zeros((len(room_data), NUM_OPS), dtype=np.int8)
hidden_matrix = np.zeros((len(room_data), NUM_OPS), dtype=np.int8)
room_types = []

for idx, row in enumerate(room_data.iter_rows(named=True)):
    visible = row['visible_ops'] or []
    hidden = row['hidden_ops'] or []
    room_cluster = row['room_cluster']

    for op in visible:
        if op < NUM_OPS:
            visible_matrix[idx, op] = 1

    for op in hidden:
        if op < NUM_OPS:
            hidden_matrix[idx, op] = 1

    room_types.append(room_cluster)

print(f"Visible matrix shape: {visible_matrix.shape}")
print(f"Hidden matrix shape: {hidden_matrix.shape}")
print(f"Sparsity (visible): {(1 - np.count_nonzero(visible_matrix) / visible_matrix.size) * 100:.1f}%")
print(f"Sparsity (hidden): {(1 - np.count_nonzero(hidden_matrix) / hidden_matrix.size) * 100:.1f}%")

# ============================================================================
# 3. CALCULATE CORRELATIONS
# ============================================================================
print("\n[3/5] Calculating operation correlations...")

# For each potential hidden operation, calculate correlation with each visible operation
# We'll use point-biserial correlation (correlation between binary variables)

# Simple approach: for each pair (visible_op, hidden_op), calculate:
# - How often they co-occur when both are present
# - Lift: P(hidden | visible) / P(hidden)

correlations = {}  # (vis_op, hid_op) -> correlation score
operation_counts = hidden_matrix.sum(axis=0)  # How common is each operation
total_rooms = len(room_data)

print("Computing pairwise correlations...")

# For efficiency, only compute for operations that appear enough times
min_count = 10
frequent_hidden_ops = np.where(operation_counts >= min_count)[0]
frequent_visible_ops = np.where(visible_matrix.sum(axis=0) >= min_count)[0]

print(f"Frequent hidden ops: {len(frequent_hidden_ops)}/{NUM_OPS}")
print(f"Frequent visible ops: {len(frequent_visible_ops)}/{NUM_OPS}")

for vis_op in frequent_visible_ops:
    vis_present = visible_matrix[:, vis_op]

    for hid_op in frequent_hidden_ops:
        hid_present = hidden_matrix[:, hid_op]

        # Calculate lift: P(hidden=1 | visible=1) / P(hidden=1)
        p_hidden = hid_present.sum() / total_rooms

        if p_hidden == 0:
            continue

        # P(hidden=1 | visible=1)
        both_present = (vis_present & hid_present).sum()
        vis_count = vis_present.sum()

        if vis_count == 0:
            continue

        p_hidden_given_visible = both_present / vis_count

        # Lift score
        lift = p_hidden_given_visible / p_hidden if p_hidden > 0 else 0

        # Only store if there's a positive correlation
        if lift > 1.0 and both_present >= 5:  # Need at least 5 co-occurrences
            correlations[(vis_op, hid_op)] = {
                'lift': lift,
                'p_hidden_given_visible': p_hidden_given_visible,
                'support': both_present / total_rooms,
                'count': both_present
            }

print(f"Found {len(correlations)} significant correlations")

# Sort by lift to see strongest patterns
top_correlations = sorted(correlations.items(), key=lambda x: x[1]['lift'], reverse=True)[:10]
print("\nTop 10 strongest correlations:")
code_to_wo = train_dataset.work_operations_dataset.code_to_wo
for (vis_op, hid_op), stats in top_correlations:
    vis_name = code_to_wo.get(vis_op, f"Op{vis_op}")[:40]
    hid_name = code_to_wo.get(hid_op, f"Op{hid_op}")[:40]
    print(f"  [{vis_op:3d}→{hid_op:3d}] Lift={stats['lift']:.2f}, "
          f"P(H|V)={stats['p_hidden_given_visible']:.3f}, Count={stats['count']}")
    print(f"    {vis_name} → {hid_name}")

# ============================================================================
# 4. BUILD ROOM-SPECIFIC PATTERNS
# ============================================================================
print("\n[4/5] Building room-specific patterns...")

# Track which operations are common for each room type
room_type_stats = defaultdict(lambda: {'visible': defaultdict(int), 'hidden': defaultdict(int), 'total': 0})

for idx, (row, room_type) in enumerate(zip(room_data.iter_rows(named=True), room_types)):
    room_type_stats[room_type]['total'] += 1

    visible = row['visible_ops'] or []
    hidden = row['hidden_ops'] or []

    for op in visible:
        room_type_stats[room_type]['visible'][op] += 1

    for op in hidden:
        room_type_stats[room_type]['hidden'][op] += 1

# Calculate room-specific base rates
room_empty_rate = {}
for room_type in room_type_stats:
    total = room_type_stats[room_type]['total']
    has_hidden = sum(1 for idx, rt in enumerate(room_types) if rt == room_type and hidden_matrix[idx].sum() > 0)
    room_empty_rate[room_type] = 1 - (has_hidden / total)

print("\nEmpty rate by room type:")
for room_type, rate in sorted(room_empty_rate.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {room_type}: {rate:.1%}")

# ============================================================================
# 5. PREDICTION FUNCTION
# ============================================================================
print("\n[5/5] Creating prediction function...")

def predict_operations(visible_ops, room_cluster, min_lift=1.5, min_confidence=0.15, max_preds=8):
    """
    Predict hidden operations based on correlations.

    Args:
        visible_ops: List of visible operation codes
        room_cluster: Room type
        min_lift: Minimum lift score to consider
        min_confidence: Minimum P(hidden|visible) to predict
        max_preds: Maximum predictions to return

    Returns:
        List of predicted operation codes
    """

    # If room type tends to be empty, be very conservative
    if room_empty_rate.get(room_cluster, 0) > 0.7:
        return []

    # Accumulate scores for each potential hidden operation
    scores = defaultdict(lambda: {'max_lift': 0, 'sum_prob': 0, 'count': 0})

    for vis_op in visible_ops:
        # Look for correlations with this visible operation
        for (v_op, h_op), stats in correlations.items():
            if v_op == vis_op and h_op not in visible_ops:
                # Update scores
                scores[h_op]['max_lift'] = max(scores[h_op]['max_lift'], stats['lift'])
                scores[h_op]['sum_prob'] += stats['p_hidden_given_visible']
                scores[h_op]['count'] += 1

    # Filter and rank predictions
    predictions = []

    for h_op, stats in scores.items():
        # Need good lift and confidence
        if stats['max_lift'] >= min_lift:
            avg_prob = stats['sum_prob'] / stats['count']

            if avg_prob >= min_confidence:
                # Combined score: lift * average probability
                combined_score = stats['max_lift'] * avg_prob
                predictions.append((h_op, combined_score, avg_prob))

    # Sort by combined score
    predictions.sort(key=lambda x: x[1], reverse=True)

    # Return top predictions
    return [op for op, _, _ in predictions[:max_preds]]

# ============================================================================
# 6. EVALUATE
# ============================================================================
print("\n[6/6] Evaluating on validation set...")

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

val_preds = []
val_targets = []

for row in val_room_data.iter_rows(named=True):
    visible = row['visible_ops'] or []
    hidden = row['hidden_ops'] or []
    room_cluster = row['room_cluster']

    pred = predict_operations(visible, room_cluster, min_lift=1.5, min_confidence=0.2, max_preds=6)
    val_preds.append(pred)
    val_targets.append(hidden if hidden else [])

val_score = normalized_rooms_score(val_preds, val_targets)
print(f"\nValidation Score: {val_score:.4f}")

# Statistics
total_preds = sum(len(p) for p in val_preds)
avg_preds = total_preds / len(val_preds)
empty_preds = sum(1 for p in val_preds if len(p) == 0)

print(f"Average predictions per room: {avg_preds:.2f}")
print(f"Empty predictions: {empty_preds}/{len(val_preds)} ({empty_preds/len(val_preds):.1%})")

# Analyze some predictions
print("\nSample predictions:")
for i in range(min(3, len(val_preds))):
    if len(val_preds[i]) > 0 or len(val_targets[i]) > 0:
        print(f"\nRoom {i}:")
        print(f"  Predicted: {val_preds[i][:5]}")
        print(f"  Actual: {val_targets[i][:5]}")
        print(f"  Match: {set(val_preds[i]) & set(val_targets[i])}")

# ============================================================================
# 7. GENERATE TEST PREDICTIONS
# ============================================================================
print("\n[7/7] Generating test predictions...")

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

    pred = predict_operations(visible, room_cluster, min_lift=1.5, min_confidence=0.2, max_preds=6)
    test_predictions[room_id] = pred

# Create submission
test_dataset.create_submission(test_predictions)

print("\n" + "="*80)
print("CORRELATION-BASED BASELINE COMPLETE!")
print("="*80)
print(f"\nValidation Score: {val_score:.4f}")
print(f"\nHow this model works:")
print("1. Builds co-occurrence matrix of all operations")
print("2. Calculates lift scores: P(hidden|visible) / P(hidden)")
print("3. For each room, predicts operations with high lift from visible ops")
print("4. Uses room type to adjust predictions")
print("5. Filters by minimum confidence and lift thresholds")
print("\nKey parameters:")
print(f"  - Minimum lift: 1.5 (operation must be 1.5x more likely given visible op)")
print(f"  - Minimum confidence: 0.2 (20% chance given visible op)")
print(f"  - Maximum predictions: 6 per room")
