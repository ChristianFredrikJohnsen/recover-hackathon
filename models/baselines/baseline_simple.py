"""
Ultra-Simple Rule-Based Baseline

This script creates a minimal baseline without any ML dependencies.
Approach:
1. Learn co-occurrence patterns from training data
2. For each test room, predict operations that frequently co-occur with visible ones
3. Weight by room type
"""

import os
import sys
import numpy as np
import polars as pl
from collections import defaultdict, Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dataset.hackathon import HackathonDataset
from metrics import normalized_rooms_score

print("="*80)
print("SIMPLE RULE-BASED BASELINE")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/4] Loading datasets...")

train_dataset = HackathonDataset(split="train", download=False, seed=42, root="data", fraction=0.15)
val_dataset = HackathonDataset(split="val", download=False, seed=42, root="data")
test_dataset = HackathonDataset(split="test", download=False, seed=42, root="data")

print(f"Train size: {len(train_dataset)}")
print(f"Val size: {len(val_dataset)}")
print(f"Test size: {len(test_dataset)}")

# ============================================================================
# 2. LEARN CO-OCCURRENCE PATTERNS
# ============================================================================
print("\n[2/4] Learning co-occurrence patterns...")

train_df = train_dataset.get_polars_dataframe()

# Get all operations per room (only visible ones)
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

# Join to get both visible and hidden for each room
room_data = room_visible.join(
    room_hidden,
    on=["project_id", "room"],
    how="left"
)

# Build co-occurrence matrix: if op A is visible, how often is op B hidden?
co_occurrence = defaultdict(lambda: defaultdict(int))
total_rooms = defaultdict(int)

# Also track by room type
room_co_occurrence = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
room_totals = defaultdict(lambda: defaultdict(int))

for row in room_data.iter_rows(named=True):
    visible = row['visible_ops'] or []
    hidden = row['hidden_ops'] or []
    room_cluster = row['room_cluster']

    for vis_op in visible:
        total_rooms[vis_op] += 1
        room_totals[room_cluster][vis_op] += 1

        for hid_op in hidden:
            co_occurrence[vis_op][hid_op] += 1
            room_co_occurrence[room_cluster][vis_op][hid_op] += 1

print(f"Learned patterns from {len(room_data)} rooms")
print(f"Tracked {len(co_occurrence)} visible operations")

# ============================================================================
# 3. CREATE PREDICTION FUNCTION
# ============================================================================
print("\n[3/4] Creating prediction function...")

def predict_missing_operations(visible_ops, room_cluster, top_n=10, use_room_specific=True):
    """
    Predict missing operations based on co-occurrence patterns.

    Args:
        visible_ops: List of visible operation codes
        room_cluster: Room type
        top_n: Number of top predictions to return
        use_room_specific: Whether to use room-specific patterns

    Returns:
        List of predicted operation codes
    """

    # Count votes for each hidden operation
    votes = defaultdict(float)

    for vis_op in visible_ops:
        # Use room-specific patterns if available
        if use_room_specific and room_cluster in room_co_occurrence and vis_op in room_co_occurrence[room_cluster]:
            patterns = room_co_occurrence[room_cluster][vis_op]
            total = room_totals[room_cluster].get(vis_op, 1)
        # Fallback to global patterns
        elif vis_op in co_occurrence:
            patterns = co_occurrence[vis_op]
            total = total_rooms.get(vis_op, 1)
        else:
            continue

        # Vote for each hidden operation weighted by co-occurrence probability
        for hid_op, count in patterns.items():
            # Don't predict operations that are already visible
            if hid_op not in visible_ops:
                probability = count / total
                votes[hid_op] += probability

    # Sort by votes and return top N
    sorted_preds = sorted(votes.items(), key=lambda x: x[1], reverse=True)
    predictions = [op for op, score in sorted_preds[:top_n]]

    return predictions

# ============================================================================
# 4. EVALUATE AND PREDICT
# ============================================================================
print("\n[4/4] Generating predictions...")

# Validation
print("\nValidating on validation set...")
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

val_room_data = val_room_visible.join(
    val_room_hidden,
    on=["project_id", "room"],
    how="left"
)

val_preds = []
val_targets = []

for row in val_room_data.iter_rows(named=True):
    visible = row['visible_ops'] or []
    hidden = row['hidden_ops'] or []
    room_cluster = row['room_cluster']

    pred = predict_missing_operations(visible, room_cluster, top_n=10)
    val_preds.append(pred)
    val_targets.append(hidden if hidden else [])

val_score = normalized_rooms_score(val_preds, val_targets)
print(f"Validation Score: {val_score:.4f}")

# Test predictions
print("\nGenerating test predictions...")
test_df = test_dataset.get_polars_dataframe()

test_room_data = test_df.group_by(["project_id", "room"]).agg([
    pl.col("work_operation").alias("visible_ops"),
    pl.col("room_cluster").first(),
    pl.col("index").first().alias("room_id"),  # Use index as room ID
])

test_predictions = {}

for row in test_room_data.iter_rows(named=True):
    visible = row['visible_ops'] or []
    room_cluster = row['room_cluster']
    room_id = row['room_id']

    pred = predict_missing_operations(visible, room_cluster, top_n=10)
    test_predictions[room_id] = pred

print(f"Generated predictions for {len(test_predictions)} rooms")

# Create submission
test_dataset.create_submission(test_predictions)

print("\n" + "="*80)
print("BASELINE COMPLETE!")
print("="*80)
print(f"\nValidation Score: {val_score:.4f}")
print(f"Predicted for {len(test_predictions)} test rooms")
print("\nSubmission file created in submissions/ folder")
print("\nHow this baseline works:")
print("1. Learns which operations often co-occur in training data")
print("2. For each test room, looks at visible operations")
print("3. Predicts operations that frequently appear together with visible ones")
print("4. Uses room-specific patterns when available")
print("\nImprovements to try:")
print("- Use machine learning instead of simple counting")
print("- Consider context from other rooms in the project")
print("- Weight rare operations differently")
print("- Tune the top_n parameter")
