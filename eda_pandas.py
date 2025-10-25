"""
Simple EDA using get_pandas_dataframe() from HackathonDataset
This explores the tabular representation of the data for building GPU-accelerated models
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from dataset.hackathon import HackathonDataset

print("="*80)
print("PANDAS-BASED EDA FOR GPU MODEL DEVELOPMENT")
print("="*80)

# ============================================================================
# 1. Load datasets and get pandas dataframes
# ============================================================================
print("\n" + "="*80)
print("1. LOADING DATA")
print("="*80)

print("\n--- Loading datasets ---")
train_dataset = HackathonDataset(split="train", download=False, seed=42, root="data", fraction=0.2)
val_dataset = HackathonDataset(split="val", download=False, seed=42, root="data")
test_dataset = HackathonDataset(split="test", download=False, seed=42, root="data")

print(f"Train dataset length (20% sample): {len(train_dataset)}")
print(f"Val dataset length: {len(val_dataset)}")
print(f"Test dataset length: {len(test_dataset)}")

print("\n--- Converting to pandas dataframes ---")
train_df = train_dataset.get_pandas_dataframe()
val_df = val_dataset.get_pandas_dataframe()
test_df = test_dataset.get_pandas_dataframe()

print(f"Train dataframe shape: {train_df.shape}")
print(f"Val dataframe shape: {val_df.shape}")
print(f"Test dataframe shape: {test_df.shape}")

# ============================================================================
# 2. Explore dataframe structure
# ============================================================================
print("\n" + "="*80)
print("2. DATAFRAME STRUCTURE")
print("="*80)

print("\n--- Column names and types ---")
print(train_df.dtypes)

print("\n--- First few rows ---")
print(train_df.head(10))

print("\n--- Unique value counts ---")
print(f"Unique rooms (index): {train_df['index'].nunique()}")
print(f"Unique projects: {train_df['project_id'].nunique()}")
print(f"Unique room types: {train_df['room'].nunique()}")
print(f"Unique room clusters: {train_df['room_cluster'].nunique()}")
print(f"Unique work operations: {train_df['work_operation'].nunique()}")

print("\n--- Room clusters ---")
print(train_df['room_cluster'].value_counts())

print("\n--- Insurance companies ---")
print(train_df['insurance_company'].value_counts())

# ============================================================================
# 3. is_hidden distribution (key for understanding the task)
# ============================================================================
print("\n" + "="*80)
print("3. VISIBLE vs HIDDEN OPERATIONS")
print("="*80)

print("\n--- is_hidden distribution ---")
print(train_df['is_hidden'].value_counts())
print(f"\nPercentage hidden: {100 * train_df['is_hidden'].sum() / len(train_df):.2f}%")

print("\n--- Operations per room (by visibility) ---")
visible_ops_per_room = train_df[~train_df['is_hidden']].groupby('index').size()
hidden_ops_per_room = train_df[train_df['is_hidden']].groupby('index').size()

print(f"Avg visible operations per room: {visible_ops_per_room.mean():.2f} (±{visible_ops_per_room.std():.2f})")
print(f"Avg hidden operations per room: {hidden_ops_per_room.mean():.2f} (±{hidden_ops_per_room.std():.2f})")

# ============================================================================
# 4. Work operations analysis
# ============================================================================
print("\n" + "="*80)
print("4. WORK OPERATIONS ANALYSIS")
print("="*80)

print("\n--- Most common visible operations ---")
visible_ops = train_df[~train_df['is_hidden']]['work_operation'].value_counts().head(20)
print(visible_ops)

print("\n--- Most common hidden operations ---")
hidden_ops = train_df[train_df['is_hidden']]['work_operation'].value_counts().head(20)
print(hidden_ops)

print("\n--- Operations that appear most in both visible and hidden ---")
common_visible = set(visible_ops.head(50).index)
common_hidden = set(hidden_ops.head(50).index)
overlap = common_visible & common_hidden
print(f"Overlap in top 50: {len(overlap)} operations")
print(f"Examples: {list(overlap)[:10]}")

# ============================================================================
# 5. Room-specific patterns
# ============================================================================
print("\n" + "="*80)
print("5. ROOM-SPECIFIC PATTERNS")
print("="*80)

print("\n--- Operations per room type (mean) ---")
ops_by_room = train_df.groupby(['room_cluster', 'is_hidden']).size().unstack(fill_value=0)
ops_by_room['total'] = ops_by_room.sum(axis=1)
ops_by_room['pct_hidden'] = 100 * ops_by_room.get(True, 0) / ops_by_room['total']
print(ops_by_room.sort_values('total', ascending=False))

# ============================================================================
# 6. Metadata features
# ============================================================================
print("\n" + "="*80)
print("6. METADATA FEATURES")
print("="*80)

print("\n--- Case creation year distribution ---")
print(train_df['case_creation_year'].value_counts().sort_index())

print("\n--- Case creation month distribution ---")
print(train_df['case_creation_month'].value_counts().sort_index())

print("\n--- Office distance statistics ---")
print(f"Mean: {train_df['office_distance'].mean():.2f} km")
print(f"Median: {train_df['office_distance'].median():.2f} km")
print(f"Std: {train_df['office_distance'].std():.2f} km")
print(f"Min: {train_df['office_distance'].min():.2f} km")
print(f"Max: {train_df['office_distance'].max():.2f} km")

# ============================================================================
# 7. Prepare feature matrix structure (for GPU model)
# ============================================================================
print("\n" + "="*80)
print("7. FEATURE MATRIX STRUCTURE")
print("="*80)

print("\n--- Room-level aggregation ---")
# Group by room (index) to understand room-level structure
room_features = train_df.groupby('index').agg({
    'project_id': 'first',
    'room': 'first',
    'room_cluster': 'first',
    'work_operation': 'count',  # total operations
    'is_hidden': 'sum',  # number of hidden operations
    'insurance_company': 'first',
    'case_creation_year': 'first',
    'case_creation_month': 'first',
    'office_distance': 'first',
}).rename(columns={'work_operation': 'total_ops', 'is_hidden': 'n_hidden'})

room_features['n_visible'] = room_features['total_ops'] - room_features['n_hidden']
room_features['pct_hidden'] = 100 * room_features['n_hidden'] / room_features['total_ops']

print(f"Room-level feature shape: {room_features.shape}")
print("\n--- Sample room features ---")
print(room_features.head(10))

print("\n--- Room statistics ---")
print(f"Rooms with 0 hidden ops: {(room_features['n_hidden'] == 0).sum()}")
print(f"Rooms with all hidden ops: {(room_features['n_hidden'] == room_features['total_ops']).sum()}")
print(f"Mean % hidden per room: {room_features['pct_hidden'].mean():.2f}%")

# ============================================================================
# 8. Co-occurrence matrix structure
# ============================================================================
print("\n" + "="*80)
print("8. CO-OCCURRENCE PATTERNS")
print("="*80)

print("\n--- Creating operation co-occurrence matrix ---")
# Get visible and hidden operations for each room
visible_room_ops = train_df[~train_df['is_hidden']].groupby('index')['work_operation'].apply(list)
hidden_room_ops = train_df[train_df['is_hidden']].groupby('index')['work_operation'].apply(list)

print(f"Rooms with visible operations: {len(visible_room_ops)}")
print(f"Rooms with hidden operations: {len(hidden_room_ops)}")

# Sample a few rooms to show structure
sample_indices = room_features.sample(5, random_state=42).index
for idx in sample_indices:
    visible = visible_room_ops.get(idx, [])
    hidden = hidden_room_ops.get(idx, [])
    print(f"\nRoom {idx}:")
    print(f"  Visible ({len(visible)}): {visible[:5]}..." if len(visible) > 5 else f"  Visible: {visible}")
    print(f"  Hidden ({len(hidden)}): {hidden[:5]}..." if len(hidden) > 5 else f"  Hidden: {hidden}")

# ============================================================================
# 9. Test set structure (for submission)
# ============================================================================
print("\n" + "="*80)
print("9. TEST SET STRUCTURE")
print("="*80)

print("\n--- Test set analysis ---")
print(f"Test dataframe shape: {test_df.shape}")
print(f"Unique test rooms: {test_df['index'].nunique()}")
print(f"Test operations per room: {test_df.groupby('index').size().describe()}")

print("\n--- Test set metadata ---")
print(f"Test projects: {test_df['project_id'].nunique()}")
print(f"Test room types: {test_df['room_cluster'].value_counts()}")

print("\n--- Test is_hidden distribution ---")
# In test set, all should be visible (is_hidden=False)
print(test_df['is_hidden'].value_counts())
if test_df['is_hidden'].all() == False:
    print("\n✓ Confirmed: All test operations are visible (as expected)")
else:
    print("\n⚠ Warning: Some test operations marked as hidden")

# ============================================================================
# 10. Summary for GPU model development
# ============================================================================
print("\n" + "="*80)
print("10. SUMMARY FOR GPU MODEL")
print("="*80)

print(f"""
KEY INSIGHTS FOR GPU MODEL:
---------------------------
✓ Data Structure:
  - Each row = one operation in one room
  - Rooms identified by 'index' column
  - Operations can be visible (X) or hidden (Y)
  - Metadata columns: insurance_company, year, month, office_distance

✓ Feature Engineering:
  - Need to pivot operations into one-hot encoded feature matrix
  - Room cluster (11 categories) should be one-hot encoded
  - Insurance company (14 categories) already has one-hot encoding column
  - Temporal features: year ({train_df['case_creation_year'].min()}-{train_df['case_creation_year'].max()}), month (1-12)

✓ Target Variable:
  - Multi-label classification: predict which operations are hidden
  - {train_df['work_operation'].nunique()} possible operation classes
  - Average {hidden_ops_per_room.mean():.1f} hidden operations per room
  - Can have 0 hidden operations (empty room)

✓ Model Architecture Considerations:
  - Input: sparse binary matrix (visible operations) + metadata features
  - Output: multi-label binary predictions (388-dimensional)
  - Can use binary cross-entropy loss
  - Consider class imbalance (some operations very rare)

✓ GPU Acceleration:
  - Convert pandas dataframe to dense/sparse tensors
  - Batch processing for efficiency
  - Can use PyTorch DataLoader with custom collate function

✓ Submission Format:
  - Predict for {test_df['index'].nunique()} rooms
  - Each prediction: room_id -> list of operation codes
  - Use dataset.create_submission() method
""")

print("\n" + "="*80)
print("EDA COMPLETE - READY FOR GPU MODEL DEVELOPMENT")
print("="*80)
