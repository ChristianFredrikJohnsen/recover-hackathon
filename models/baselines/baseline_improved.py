"""
Improved Rule-Based Baseline

Key improvements:
1. More conservative predictions (higher threshold)
2. Better handling of empty cases
3. Consider only strong co-occurrence patterns
"""

import os
import sys
import numpy as np
import polars as pl
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dataset.hackathon import HackathonDataset
from metrics import normalized_rooms_score

print("="*80)
print("IMPROVED RULE-BASED BASELINE")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/4] Loading datasets...")

train_dataset = HackathonDataset(split="train", download=False, seed=42, root="data", fraction=0.2)
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

# Get all operations per room
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

# Build conditional probability: P(hidden_op | visible_op)
conditional_prob = defaultdict(lambda: defaultdict(lambda: {"count": 0, "total": 0}))
room_conditional_prob = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"count": 0, "total": 0})))

# Also track when rooms have NO hidden operations
empty_rooms_by_type = defaultdict(int)
total_rooms_by_type = defaultdict(int)

for row in room_data.iter_rows(named=True):
    visible = row['visible_ops'] or []
    hidden = row['hidden_ops'] or []
    room_cluster = row['room_cluster']

    total_rooms_by_type[room_cluster] += 1

    if not hidden:
        empty_rooms_by_type[room_cluster] += 1
        continue

    for vis_op in visible:
        # Global statistics
        conditional_prob[vis_op]["total"]["total"] += 1

        # Room-specific statistics
        room_conditional_prob[room_cluster][vis_op]["total"]["total"] += 1

        for hid_op in hidden:
            conditional_prob[vis_op][hid_op]["count"] += 1
            room_conditional_prob[room_cluster][vis_op][hid_op]["count"] += 1

print(f"Learned patterns from {len(room_data)} rooms")

# Calculate probability of empty rooms per type
empty_prob = {}
for room_type in empty_rooms_by_type:
    empty_prob[room_type] = empty_rooms_by_type[room_type] / total_rooms_by_type[room_type]

print(f"\nProbability of empty predictions by room type:")
for room_type, prob in sorted(empty_prob.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {room_type}: {prob:.2%}")

# ============================================================================
# 3. PREDICTION FUNCTION
# ============================================================================
print("\n[3/4] Creating prediction function...")

def predict_missing_operations(visible_ops, room_cluster, min_confidence=0.15, max_predictions=5):
    """
    More conservative prediction strategy.

    Args:
        visible_ops: List of visible operation codes
        room_cluster: Room type
        min_confidence: Minimum P(hidden | visible) to make a prediction
        max_predictions: Maximum number of predictions

    Returns:
        List of predicted operation codes
    """

    # If room type tends to have many empty predictions, be conservative
    if empty_prob.get(room_cluster, 0) > 0.5:
        return []

    votes = defaultdict(float)

    for vis_op in visible_ops:
        # Try room-specific patterns first
        if room_cluster in room_conditional_prob and vis_op in room_conditional_prob[room_cluster]:
            patterns = room_conditional_prob[room_cluster][vis_op]
            total = patterns["total"]["total"]

            if total >= 5:  # Need at least 5 examples
                for hid_op, stats in patterns.items():
                    if hid_op == "total":
                        continue

                    prob = stats["count"] / total

                    # Only vote if probability is high enough
                    if prob >= min_confidence and hid_op not in visible_ops:
                        votes[hid_op] += prob

        # Fallback to global patterns
        elif vis_op in conditional_prob:
            patterns = conditional_prob[vis_op]
            total = patterns["total"]["total"]

            if total >= 10:  # Need more examples for global patterns
                for hid_op, stats in patterns.items():
                    if hid_op == "total":
                        continue

                    prob = stats["count"] / total

                    if prob >= min_confidence and hid_op not in visible_ops:
                        votes[hid_op] += prob * 0.7  # Weight global patterns less

    # Sort by confidence and return top predictions
    sorted_preds = sorted(votes.items(), key=lambda x: x[1], reverse=True)

    # Only return predictions if we have strong signals
    strong_preds = [(op, score) for op, score in sorted_preds if score >= min_confidence]

    predictions = [op for op, _ in strong_preds[:max_predictions]]

    return predictions

# ============================================================================
# 4. EVALUATE
# ============================================================================
print("\n[4/4] Evaluating and generating predictions...")

# Validation
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

    pred = predict_missing_operations(visible, room_cluster, min_confidence=0.2, max_predictions=5)
    val_preds.append(pred)
    val_targets.append(hidden if hidden else [])

val_score = normalized_rooms_score(val_preds, val_targets)
print(f"\nValidation Score: {val_score:.4f}")

# Analyze predictions
total_preds = sum(len(p) for p in val_preds)
avg_preds = total_preds / len(val_preds)
empty_preds = sum(1 for p in val_preds if len(p) == 0)
print(f"Average predictions per room: {avg_preds:.2f}")
print(f"Empty predictions: {empty_preds}/{len(val_preds)} ({empty_preds/len(val_preds):.1%})")

# Test predictions
print("\nGenerating test predictions...")
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

    pred = predict_missing_operations(visible, room_cluster, min_confidence=0.2, max_predictions=5)
    test_predictions[room_id] = pred

# Create submission
test_dataset.create_submission(test_predictions)

print("\n" + "="*80)
print("IMPROVED BASELINE COMPLETE!")
print("="*80)
print(f"\nValidation Score: {val_score:.4f}")
print("\nKey improvements:")
print("1. More conservative predictions (higher confidence threshold)")
print("2. Predict empty lists for room types that often have no missing ops")
print("3. Weight room-specific patterns higher than global patterns")
print("4. Require minimum number of training examples")
