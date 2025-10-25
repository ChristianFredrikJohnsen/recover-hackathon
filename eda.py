"""
Exploratory Data Analysis Script for Recover Hackathon Dataset
This script performs comprehensive EDA on the dataset to understand:
- Data structure and formats
- Distribution of work operations, rooms, and metadata
- Relationships between features
- Missing values and data quality
"""

import os
import sys
import numpy as np
import pandas as pd
import polars as pl
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.hackathon import HackathonDataset
from dataset.work_operations import WorkOperationsDataset
from dataset.metadata import MetadataDataset

print("="*80)
print("RECOVER HACKATHON - EXPLORATORY DATA ANALYSIS")
print("="*80)

# ============================================================================
# 1. RAW CSV DATA EXPLORATION
# ============================================================================
print("\n" + "="*80)
print("1. RAW CSV DATA EXPLORATION")
print("="*80)

data_folder = "data"

# Load raw CSV files
print("\n--- Loading Raw CSV Files ---")
train_df = pl.read_csv(os.path.join(data_folder, "train.csv"))
val_df = pl.read_csv(os.path.join(data_folder, "val.csv"))
test_df = pl.read_csv(os.path.join(data_folder, "test.csv"))
metadata_df = pl.read_csv(os.path.join(data_folder, "metaData.csv"))
tickets_df = pl.read_csv(os.path.join(data_folder, "tickets.csv"))

print(f"\nTrain shape: {train_df.shape}")
print(f"Val shape: {val_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Metadata shape: {metadata_df.shape}")
print(f"Tickets shape: {tickets_df.shape}")

# Show columns
print("\n--- Train CSV Columns ---")
print(train_df.columns)
print("\n--- Sample rows from train.csv ---")
print(train_df.head())

print("\n--- Metadata CSV Columns ---")
print(metadata_df.columns)
print("\n--- Sample rows from metaData.csv ---")
print(metadata_df.head())

print("\n--- Tickets CSV Columns ---")
print(tickets_df.columns)
print("\n--- Sample rows from tickets.csv ---")
print(tickets_df.head())

# Check for missing values in raw data
print("\n--- Missing Values in Raw Data ---")
print("Train missing values:")
print(train_df.null_count())
print("\nMetadata missing values:")
print(metadata_df.null_count())

# ============================================================================
# 2. WORK OPERATIONS ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("2. WORK OPERATIONS ANALYSIS")
print("="*80)

# Count unique values
print("\n--- Unique Counts ---")
print(f"Unique projects in train: {train_df['project_id'].n_unique()}")
print(f"Unique projects in val: {val_df['project_id'].n_unique()}")
print(f"Unique projects in test: {test_df['project_id'].n_unique()}")
print(f"Unique rooms in train: {train_df.select(pl.concat_str(['project_id', 'room'])).n_unique()}")
print(f"Unique work operation clusters: {train_df['work_operation_cluster_code'].n_unique()}")
print(f"Unique work operation names: {train_df['work_operation_cluster_name'].n_unique()}")

# Work operation distribution
print("\n--- Top 20 Most Common Work Operations ---")
wo_counts = train_df.group_by("work_operation_cluster_name").agg(
    pl.len().alias("count")
).sort("count", descending=True)
print(wo_counts.head(20))

print("\n--- Work Operation Cluster Code Range ---")
print(f"Min code: {train_df['work_operation_cluster_code'].min()}")
print(f"Max code: {train_df['work_operation_cluster_code'].max()}")
print(f"Total unique codes: {train_df['work_operation_cluster_code'].n_unique()}")

# Room distribution
print("\n--- Room Distribution ---")
room_counts = train_df.group_by("room").agg(
    pl.len().alias("count")
).sort("count", descending=True)
print(room_counts.head(20))

# Operations per room statistics
print("\n--- Operations per Room Statistics ---")
ops_per_room = train_df.group_by(["project_id", "room"]).agg(
    pl.len().alias("n_operations")
)
print(f"Mean operations per room: {ops_per_room['n_operations'].mean():.2f}")
print(f"Median operations per room: {ops_per_room['n_operations'].median():.2f}")
print(f"Min operations per room: {ops_per_room['n_operations'].min()}")
print(f"Max operations per room: {ops_per_room['n_operations'].max()}")
print(f"Std operations per room: {ops_per_room['n_operations'].std():.2f}")

# Rooms per project
print("\n--- Rooms per Project Statistics ---")
rooms_per_project = train_df.group_by("project_id").agg(
    pl.col("room").n_unique().alias("n_rooms")
)
print(f"Mean rooms per project: {rooms_per_project['n_rooms'].mean():.2f}")
print(f"Median rooms per project: {rooms_per_project['n_rooms'].median():.2f}")
print(f"Min rooms per project: {rooms_per_project['n_rooms'].min()}")
print(f"Max rooms per project: {rooms_per_project['n_rooms'].max()}")

# ============================================================================
# 3. METADATA ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("3. METADATA ANALYSIS")
print("="*80)

print("\n--- Insurance Companies ---")
print(f"Unique insurance companies: {metadata_df['insurance_company'].n_unique()}")
company_dist = metadata_df.group_by("insurance_company").agg(
    pl.len().alias("count")
).sort("count", descending=True)
print(company_dist)

print("\n--- Case Creation Year Distribution ---")
year_dist = metadata_df.group_by("case_creation_year").agg(
    pl.len().alias("count")
).sort("case_creation_year")
print(year_dist)

print("\n--- Case Creation Month Distribution ---")
month_dist = metadata_df.group_by("case_creation_month").agg(
    pl.len().alias("count")
).sort("case_creation_month")
print(month_dist)

print("\n--- Office Distance Statistics ---")
# Clean and convert office_distance
office_dist = metadata_df.with_columns(
    pl.col("office_distance").str.replace_all(",", ".").cast(pl.Float32)
)
print(f"Mean office distance: {office_dist['office_distance'].mean():.2f} km")
print(f"Median office distance: {office_dist['office_distance'].median():.2f} km")
print(f"Min office distance: {office_dist['office_distance'].min():.2f} km")
print(f"Max office distance: {office_dist['office_distance'].max():.2f} km")

print("\n--- Zip Code Analysis ---")
print(f"Unique recover office zip codes: {metadata_df['recover_office_zip_code'].n_unique()}")
print(f"Unique damage address zip codes: {metadata_df['damage_address_zip_code'].n_unique()}")

# ============================================================================
# 4. TICKETS ANALYSIS (Work Operation Popularity)
# ============================================================================
print("\n" + "="*80)
print("4. TICKETS ANALYSIS (Work Operation Weights)")
print("="*80)

print("\n--- Top 20 Work Operations by Ticket Count ---")
top_tickets = tickets_df.sort("n_tickets", descending=True).head(20)
print(top_tickets)

print("\n--- Bottom 20 Work Operations by Ticket Count ---")
bottom_tickets = tickets_df.sort("n_tickets").head(20)
print(bottom_tickets)

print("\n--- Ticket Distribution Statistics ---")
print(f"Mean tickets: {tickets_df['n_tickets'].mean():.2f}")
print(f"Median tickets: {tickets_df['n_tickets'].median():.2f}")
print(f"Min tickets: {tickets_df['n_tickets'].min()}")
print(f"Max tickets: {tickets_df['n_tickets'].max()}")

# ============================================================================
# 5. HACKATHON DATASET EXPLORATION
# ============================================================================
print("\n" + "="*80)
print("5. HACKATHON DATASET EXPLORATION (PyTorch Dataset)")
print("="*80)

print("\n--- Loading HackathonDataset ---")
val_dataset = HackathonDataset(split="val", download=False, seed=42, root="data")
print(f"Val dataset length: {len(val_dataset)}")
test_dataset = HackathonDataset(split="test", download=False, seed=42, root="data")
print(f"Test dataset length: {len(test_dataset)}")
train_dataset = HackathonDataset(split="train", download=False, seed=42, root="data", fraction=0.15) # load only 15% of the train dataset to speed up the analysis
print(f"Train dataset length (15% sample): {len(train_dataset)}")


print("\n--- Sample Data Point ---")
sample = train_dataset[0]
print(f"Keys: {sample.keys()}")
print(f"ID: {sample['id']}")
print(f"Project ID: {sample['project_id']}")
print(f"Room cluster: {sample['room_cluster']}")
print(f"X shape (visible operations): {sample['X'].shape}")
print(f"Y shape (hidden operations): {sample['Y'].shape}")
print(f"X_codes (visible operation codes): {sample['X_codes']}")
print(f"Y_codes (hidden operation codes): {sample['Y_codes']}")
print(f"Number of visible operations: {sample['X'].sum().item()}")
print(f"Number of hidden operations: {sample['Y'].sum().item()}")
print(f"Room cluster one-hot shape: {sample['room_cluster_one_hot'].shape}")
print(f"Number of context rooms: {len(sample['calculus'])}")

# Decode operation names
print(f"\nVisible operation names: {train_dataset.to_cluster_names(sample['X_codes'])}")
print(f"Hidden operation names: {train_dataset.to_cluster_names(sample['Y_codes'])}")

# Context structure
print("\n--- Context Structure ---")
if len(sample['calculus']) > 0:
    context_sample = sample['calculus'][0]
    print(f"Context keys: {context_sample.keys()}")
    print(f"Context room: {context_sample['room']}")
    print(f"Context room cluster: {context_sample['room_cluster']}")
    print(f"Context operations: {context_sample['work_operations']}")

# ============================================================================
# 6. TABULAR DATA EXPLORATION (Pandas/Polars DataFrame)
# ============================================================================
print("\n" + "="*80)
print("6. TABULAR DATA EXPLORATION (DataFrame Format)")
print("="*80)

print("\n--- Getting Polars DataFrame ---")
df_polars = train_dataset.get_polars_dataframe()
print(f"DataFrame shape: {df_polars.shape}")
print(f"Columns: {df_polars.columns}")
print("\n--- Sample rows ---")
print(df_polars.head())

print("\n--- is_hidden Distribution ---")
hidden_dist = df_polars.group_by("is_hidden").agg(
    pl.len().alias("count")
)
print(hidden_dist)

print("\n--- Work Operation Distribution by is_hidden ---")
print(f"Total visible operations: {df_polars.filter(pl.col('is_hidden') == False).shape[0]}")
print(f"Total hidden operations: {df_polars.filter(pl.col('is_hidden') == True).shape[0]}")

# ============================================================================
# 7. ENCODING VERIFICATION
# ============================================================================
print("\n" + "="*80)
print("7. ENCODING VERIFICATION")
print("="*80)

print("\n--- Number of Clusters ---")
print(f"Number of work operation clusters: {train_dataset.work_operations_dataset.num_clusters}")

print("\n--- Room Clusters ---")
room_to_idx = train_dataset.work_operations_dataset.room_to_index
print(f"Room to index mapping: {room_to_idx}")

print("\n--- Verifying One-Hot Encoding ---")
sample_codes = [1, 5, 10]
encoded = train_dataset.multi_hot_encode_list(sample_codes)
print(f"Sample codes: {sample_codes}")
print(f"Encoded length: {len(encoded)}")
print(f"Non-zero indices: {[i for i, x in enumerate(encoded) if x == 1]}")

print("\n--- Code to Name Mapping (first 10) ---")
code_to_wo = train_dataset.work_operations_dataset.code_to_wo
for code in sorted(code_to_wo.keys())[:10]:
    print(f"Code {code}: {code_to_wo[code]}")

# ============================================================================
# 8. DATA SPLIT CONSISTENCY
# ============================================================================
print("\n" + "="*80)
print("8. DATA SPLIT CONSISTENCY")
print("="*80)

train_projects = set(train_df['project_id'].unique())
val_projects = set(val_df['project_id'].unique())
test_projects = set(test_df['project_id'].unique())

print(f"\nTrain projects: {len(train_projects)}")
print(f"Val projects: {len(val_projects)}")
print(f"Test projects: {len(test_projects)}")

overlap_train_val = train_projects & val_projects
overlap_train_test = train_projects & test_projects
overlap_val_test = val_projects & test_projects

print(f"\nOverlap train-val: {len(overlap_train_val)}")
print(f"Overlap train-test: {len(overlap_train_test)}")
print(f"Overlap val-test: {len(overlap_val_test)}")

# ============================================================================
# 9. SUMMARY AND KEY INSIGHTS
# ============================================================================
print("\n" + "="*80)
print("9. SUMMARY AND KEY INSIGHTS")
print("="*80)

print(f"""
KEY DATASET FACTS:
------------------
✓ Dataset Structure:
  - Training rooms: {len(train_dataset)}
  - Validation rooms: {len(val_dataset)}
  - Test rooms (for submission): {len(test_dataset)}
  
✓ Work Operations:
  - Total unique operation clusters: {train_df['work_operation_cluster_code'].n_unique()}
  - Cluster codes range: {train_df['work_operation_cluster_code'].min()} to {train_df['work_operation_cluster_code'].max()}
  - Operations are encoded as one-hot vectors of length {train_dataset.work_operations_dataset.num_clusters}
  
✓ Room Information:
  - {len(room_to_idx)} standardized room types: {list(room_to_idx.keys())}
  - Average operations per room: {ops_per_room['n_operations'].mean():.2f}
  - Average rooms per project: {rooms_per_project['n_rooms'].mean():.2f}
  
✓ Projects:
  - Training projects: {len(train_projects)}
  - Validation projects: {len(val_projects)}
  - Test projects: {len(test_projects)}
  - Project splits are non-overlapping: {len(overlap_train_val) == 0 and len(overlap_train_test) == 0 and len(overlap_val_test) == 0}
  
✓ Metadata Features:
  - Insurance companies: {metadata_df['insurance_company'].n_unique()}
  - Years: {metadata_df['case_creation_year'].min()} - {metadata_df['case_creation_year'].max()}
  - Office distance range: {office_dist['office_distance'].min():.2f} - {office_dist['office_distance'].max():.2f} km
  
✓ Data Format:
  - Each room has X (visible operations) and Y (hidden operations to predict)
  - X and Y are one-hot encoded vectors
  - Context from other rooms in same project available via 'calculus' field
  - Sampling strategy controls how many operations are hidden (default: 30-50%)
  
✓ Test Set:
  - Test set has all operations visible in X
  - Y is what needs to be predicted for submission
  - Must predict for {test_dataset.work_operations_dataset.data.select('id').n_unique()} unique room IDs
""")

print("\n" + "="*80)
print("EDA COMPLETE!")
print("="*80)
print("\nRecommended next steps:")
print("1. Visualize work operation co-occurrences")
print("2. Analyze room-specific operation patterns")
print("3. Examine temporal trends (if any)")
print("4. Feature engineering based on metadata")
print("5. Build baseline models to understand task difficulty")




# OUTPUT
"""
$ python3 eda.py
================================================================================
RECOVER HACKATHON - EXPLORATORY DATA ANALYSIS
================================================================================

================================================================================
1. RAW CSV DATA EXPLORATION
================================================================================

--- Loading Raw CSV Files ---

Train shape: (1908899, 5)
Val shape: (111384, 5)
Test shape: (167173, 5)
Metadata shape: (60010, 7)
Tickets shape: (369, 2)

--- Train CSV Columns ---
['id', 'project_id', 'room', 'work_operation_cluster_name', 'work_operation_cluster_code']

--- Sample rows from train.csv ---
shape: (5, 5)
┌───────┬────────────┬─────────────┬─────────────────────────────────┬─────────────────────────────┐
│ id    ┆ project_id ┆ room        ┆ work_operation_cluster_name     ┆ work_operation_cluster_code │
│ ---   ┆ ---        ┆ ---         ┆ ---                             ┆ ---                         │
│ i64   ┆ i64        ┆ str         ┆ str                             ┆ i64                         │
╞═══════╪════════════╪═════════════╪═════════════════════════════════╪═════════════════════════════╡
│ 18299 ┆ 197011     ┆ tilkomstvei ┆ Lett beskyttelse av gulv med d… ┆ 256                         │
│ 18299 ┆ 197011     ┆ tilkomstvei ┆ Byggvask                        ┆ 108                         │
│ 18299 ┆ 197011     ┆ tilkomstvei ┆ Tildekking av trappetrinn       ┆ 260                         │
│ 18300 ┆ 270183     ┆ Bar         ┆ Ny underlag for parkett/lamina… ┆ 53                          │
│ 18300 ┆ 270183     ┆ Bar         ┆ Demont gulvlist for gjenbruk    ┆ 67                          │
└───────┴────────────┴─────────────┴─────────────────────────────────┴─────────────────────────────┘

--- Metadata CSV Columns ---
['insurance_company', 'recover_office_zip_code', 'damage_address_zip_code', 'office_distance', 'case_creation_year', 'case_creation_month', 'project_id']

--- Sample rows from metaData.csv ---
shape: (5, 7)
┌───────────────────┬─────────────────────────┬─────────────────────────┬─────────────────┬────────────────────┬─────────────────────┬────────────┐
│ insurance_company ┆ recover_office_zip_code ┆ damage_address_zip_code ┆ office_distance ┆ case_creation_year ┆ case_creation_month ┆ project_id │
│ ---               ┆ ---                     ┆ ---                     ┆ ---             ┆ ---                ┆ ---                 ┆ ---        │
│ str               ┆ i64                     ┆ i64                     ┆ str             ┆ i64                ┆ i64                 ┆ i64        │
╞═══════════════════╪═════════════════════════╪═════════════════════════╪═════════════════╪════════════════════╪═════════════════════╪════════════╡
│ C                 ┆ 7327                    ┆ 7327                    ┆ 8,8             ┆ 2023               ┆ 11                  ┆ 147474     │
│ C                 ┆ 1389                    ┆ 1387                    ┆ 4,3             ┆ 2022               ┆ 6                   ┆ 147481     │
│ K                 ┆ 8400                    ┆ 8403                    ┆ 2,7             ┆ 2020               ┆ 10                  ┆ 147483     │
│ C                 ┆ 2150                    ┆ 2160                    ┆ 7               ┆ 2023               ┆ 4                   ┆ 147485     │
│ N                 ┆ 3262                    ┆ 3265                    ┆ 2,5             ┆ 2021               ┆ 3                   ┆ 147486     │
└───────────────────┴─────────────────────────┴─────────────────────────┴─────────────────┴────────────────────┴─────────────────────┴────────────┘

--- Tickets CSV Columns ---
['work_operation_cluster_code', 'n_tickets']

--- Sample rows from tickets.csv ---
shape: (5, 2)
┌─────────────────────────────┬───────────┐
│ work_operation_cluster_code ┆ n_tickets │
│ ---                         ┆ ---       │
│ i64                         ┆ i64       │
╞═════════════════════════════╪═══════════╡
│ 68                          ┆ 5         │
│ 204                         ┆ 1         │
│ 315                         ┆ 2         │
│ 138                         ┆ 8         │
│ 340                         ┆ 23        │
└─────────────────────────────┴───────────┘

--- Missing Values in Raw Data ---
Train missing values:
shape: (1, 5)
┌─────┬────────────┬──────┬─────────────────────────────┬─────────────────────────────┐
│ id  ┆ project_id ┆ room ┆ work_operation_cluster_name ┆ work_operation_cluster_code │
│ --- ┆ ---        ┆ ---  ┆ ---                         ┆ ---                         │
│ u32 ┆ u32        ┆ u32  ┆ u32                         ┆ u32                         │
╞═════╪════════════╪══════╪═════════════════════════════╪═════════════════════════════╡
│ 0   ┆ 0          ┆ 0    ┆ 0                           ┆ 0                           │
└─────┴────────────┴──────┴─────────────────────────────┴─────────────────────────────┘

Metadata missing values:
shape: (1, 7)
┌───────────────────┬─────────────────────────┬─────────────────────────┬─────────────────┬────────────────────┬─────────────────────┬────────────┐
│ insurance_company ┆ recover_office_zip_code ┆ damage_address_zip_code ┆ office_distance ┆ case_creation_year ┆ case_creation_month ┆ project_id │
│ ---               ┆ ---                     ┆ ---                     ┆ ---             ┆ ---                ┆ ---                 ┆ ---        │
│ u32               ┆ u32                     ┆ u32                     ┆ u32             ┆ u32                ┆ u32                 ┆ u32        │
╞═══════════════════╪═════════════════════════╪═════════════════════════╪═════════════════╪════════════════════╪═════════════════════╪════════════╡
│ 0                 ┆ 140                     ┆ 75                      ┆ 0               ┆ 0                  ┆ 0                   ┆ 0          │
└───────────────────┴─────────────────────────┴─────────────────────────┴─────────────────┴────────────────────┴─────────────────────┴────────────┘

================================================================================
2. WORK OPERATIONS ANALYSIS
================================================================================

--- Unique Counts ---
Unique projects in train: 51009
Unique projects in val: 3000
Unique projects in test: 5994
Unique rooms in train: 185635
Unique work operation clusters: 368
Unique work operation names: 367

--- Top 20 Most Common Work Operations ---
/home/sve/cogito/recover-hackathon/eda.py:92: DeprecationWarning: `pl.count()` is deprecated. Please use `pl.len()` instead.
(Deprecated in version 0.20.5)
  pl.count().alias("count")
shape: (20, 2)
┌─────────────────────────────────┬───────┐
│ work_operation_cluster_name     ┆ count │
│ ---                             ┆ ---   │
│ str                             ┆ u32   │
╞═════════════════════════════════╪═══════╡
│ Lett beskyttelse av gulv med d… ┆ 68603 │
│ Byggvask                        ┆ 59512 │
│ Intern flytting av innbo mello… ┆ 59020 │
│ Ny gulvlist                     ┆ 52446 │
│ Riv gulvlist                    ┆ 51740 │
│ …                               ┆ …     │
│ Ny dampsperre                   ┆ 30661 │
│ Ny feielist                     ┆ 30348 │
│ Riv feielist                    ┆ 29297 │
│ Riv dampsperre / vindsperre     ┆ 28355 │
│ Ny isolasjon i vegg             ┆ 25515 │
└─────────────────────────────────┴───────┘

--- Work Operation Cluster Code Range ---
Min code: 0
Max code: 387
Total unique codes: 368

--- Room Distribution ---
/home/sve/cogito/recover-hackathon/eda.py:104: DeprecationWarning: `pl.count()` is deprecated. Please use `pl.len()` instead.
(Deprecated in version 0.20.5)
  pl.count().alias("count")
shape: (20, 2)
┌──────────────┬────────┐
│ room         ┆ count  │
│ ---          ┆ ---    │
│ str          ┆ u32    │
╞══════════════╪════════╡
│ Kjøkken      ┆ 277802 │
│ Stue         ┆ 210554 │
│ Gang         ┆ 190262 │
│ Soverom      ┆ 179307 │
│ Bad          ┆ 105915 │
│ …            ┆ …      │
│ Stue 2       ┆ 13873  │
│ Kjøkken/stue ┆ 13385  │
│ Soverom 3    ┆ 12664  │
│ Kjellerstue  ┆ 12503  │
│ Kjøkken 2    ┆ 9390   │
└──────────────┴────────┘

--- Operations per Room Statistics ---
/home/sve/cogito/recover-hackathon/eda.py:111: DeprecationWarning: `pl.count()` is deprecated. Please use `pl.len()` instead.
(Deprecated in version 0.20.5)
  pl.count().alias("n_operations")
Mean operations per room: 10.28
Median operations per room: 8.00
Min operations per room: 1
Max operations per room: 92
Std operations per room: 9.62

--- Rooms per Project Statistics ---
Mean rooms per project: 3.64
Median rooms per project: 3.00
Min rooms per project: 1
Max rooms per project: 54

================================================================================
3. METADATA ANALYSIS
================================================================================

--- Insurance Companies ---
Unique insurance companies: 14
/home/sve/cogito/recover-hackathon/eda.py:139: DeprecationWarning: `pl.count()` is deprecated. Please use `pl.len()` instead.
(Deprecated in version 0.20.5)
  pl.count().alias("count")
shape: (14, 2)
┌───────────────────┬───────┐
│ insurance_company ┆ count │
│ ---               ┆ ---   │
│ str               ┆ u32   │
╞═══════════════════╪═══════╡
│ J                 ┆ 17958 │
│ C                 ┆ 15159 │
│ N                 ┆ 9810  │
│ K                 ┆ 5385  │
│ E                 ┆ 3883  │
│ …                 ┆ …     │
│ L                 ┆ 262   │
│ A                 ┆ 163   │
│ I                 ┆ 150   │
│ B                 ┆ 63    │
│ O                 ┆ 21    │
└───────────────────┴───────┘

--- Case Creation Year Distribution ---
/home/sve/cogito/recover-hackathon/eda.py:145: DeprecationWarning: `pl.count()` is deprecated. Please use `pl.len()` instead.
(Deprecated in version 0.20.5)
  pl.count().alias("count")
shape: (11, 2)
┌────────────────────┬───────┐
│ case_creation_year ┆ count │
│ ---                ┆ ---   │
│ i64                ┆ u32   │
╞════════════════════╪═══════╡
│ 2015               ┆ 1     │
│ 2016               ┆ 4     │
│ 2017               ┆ 31    │
│ 2018               ┆ 159   │
│ 2019               ┆ 1623  │
│ …                  ┆ …     │
│ 2021               ┆ 12827 │
│ 2022               ┆ 11922 │
│ 2023               ┆ 11606 │
│ 2024               ┆ 7735  │
│ 2025               ┆ 2976  │
└────────────────────┴───────┘

--- Case Creation Month Distribution ---
/home/sve/cogito/recover-hackathon/eda.py:151: DeprecationWarning: `pl.count()` is deprecated. Please use `pl.len()` instead.
(Deprecated in version 0.20.5)
  pl.count().alias("count")
shape: (12, 2)
┌─────────────────────┬───────┐
│ case_creation_month ┆ count │
│ ---                 ┆ ---   │
│ i64                 ┆ u32   │
╞═════════════════════╪═══════╡
│ 1                   ┆ 7159  │
│ 2                   ┆ 5297  │
│ 3                   ┆ 5281  │
│ 4                   ┆ 4493  │
│ 5                   ┆ 4343  │
│ …                   ┆ …     │
│ 8                   ┆ 5946  │
│ 9                   ┆ 4591  │
│ 10                  ┆ 4789  │
│ 11                  ┆ 4650  │
│ 12                  ┆ 4671  │
└─────────────────────┴───────┘

--- Office Distance Statistics ---
Mean office distance: 18.90 km
Median office distance: 8.90 km
Min office distance: 0.00 km
Max office distance: 8474.00 km

--- Zip Code Analysis ---
Unique recover office zip codes: 172
Unique damage address zip codes: 3290

================================================================================
4. TICKETS ANALYSIS (Work Operation Weights)
================================================================================

--- Top 20 Work Operations by Ticket Count ---
shape: (20, 2)
┌─────────────────────────────┬───────────┐
│ work_operation_cluster_code ┆ n_tickets │
│ ---                         ┆ ---       │
│ i64                         ┆ i64       │
╞═════════════════════════════╪═══════════╡
│ 368                         ┆ 68603     │
│ 369                         ┆ 34301     │
│ 371                         ┆ 22867     │
│ 374                         ┆ 17150     │
│ 366                         ┆ 3610      │
│ …                           ┆ …         │
│ 117                         ┆ 1124      │
│ 255                         ┆ 1106      │
│ 238                         ┆ 1088      │
│ 253                         ┆ 1055      │
│ 30                          ┆ 1039      │
└─────────────────────────────┴───────────┘

--- Bottom 20 Work Operations by Ticket Count ---
shape: (20, 2)
┌─────────────────────────────┬───────────┐
│ work_operation_cluster_code ┆ n_tickets │
│ ---                         ┆ ---       │
│ i64                         ┆ i64       │
╞═════════════════════════════╪═══════════╡
│ 204                         ┆ 1         │
│ 53                          ┆ 1         │
│ 76                          ┆ 1         │
│ 44                          ┆ 1         │
│ 313                         ┆ 1         │
│ …                           ┆ …         │
│ 124                         ┆ 2         │
│ 5                           ┆ 2         │
│ 62                          ┆ 2         │
│ 314                         ┆ 2         │
│ 6                           ┆ 2         │
└─────────────────────────────┴───────────┘

--- Ticket Distribution Statistics ---
Mean tickets: 600.52
Median tickets: 65.00
Min tickets: 1
Max tickets: 68603

================================================================================
5. HACKATHON DATASET EXPLORATION (PyTorch Dataset)
================================================================================

--- Loading HackathonDataset ---
Note: Loading only 15% of training data for faster EDA
Val dataset length: 10827
Test dataset length: 18299
Train dataset length (15% sample): 27425

--- Sample Data Point ---
Keys: dict_keys(['id', 'X', 'Y', 'project_id', 'room_cluster', 'room_cluster_one_hot', 'calculus', 'X_codes', 'Y_codes', 'insurance_company', 'insurance_company_one_hot', 'recover_office_zip_code', 'damage_address_zip_code', 'office_distance', 'case_creation_year', 'case_creation_month'])
ID: 93818
Project ID: 147490
Room cluster: ukjent
X shape (visible operations): torch.Size([388])
Y shape (hidden operations): torch.Size([388])
X_codes (visible operation codes): tensor([244, 250, 308, 236, 304, 221, 233, 287, 220, 218, 206, 232, 225, 213,
        214, 211, 207, 212, 208, 228, 210, 219, 303, 282, 209, 273])
Y_codes (hidden operation codes): tensor([307])
Number of visible operations: 26
Number of hidden operations: 1
Room cluster one-hot shape: torch.Size([11])
Number of context rooms: 0

Visible operation names: ['Demont takrenner', 'Remont takrenner', 'Riv bærende utvendig drager', 'Demont mønebeslag', 'Ny bærende utvendig drager', 'Ny mønebeslag', 'Riving av underkledning', 'Riv veggpanel utvendig', 'Riving sløyfe,strø', 'Riving forkantbord', 'Riving taktro', 'Riving nedløpsrør', 'Ny underkledning', 'Riv vannbord til vindskie', 'Riving vindskjeier', 'Ny vannbord til vindskie', 'Riving av stålplatetak', 'Ny forkantbord', 'Ny stålplatetak', 'Ny taktro', 'Ny vindskier', 'Ny nedløpsrør', 'Ny bærende terrassestolpe', 'Ny panel utvendig', 'Ny lekter tak', 'Ny utvendig belistning']
Hidden operation names: ['Riv bærende terrassestolper']

--- Context Structure ---

================================================================================
6. TABULAR DATA EXPLORATION (DataFrame Format)
================================================================================

--- Getting Polars DataFrame ---
DataFrame shape: (281599, 17)
Columns: ['index', 'project_id', 'room', 'room_cluster', 'work_operation', 'use_balanced_data', 'sample_pct', 'subset_size', 'use_sampled_calculus', 'is_hidden', 'insurance_company', 'recover_office_zip_code', 'damage_address_zip_code', 'office_distance', 'case_creation_year', 'case_creation_month', 'insurance_company_one_hot']

--- Sample rows ---
shape: (5, 17)
┌───────┬────────────┬─────────┬──────────────┬───┬─────────────────┬────────────────────┬─────────────────────┬───────────────────────────┐
│ index ┆ project_id ┆ room    ┆ room_cluster ┆ … ┆ office_distance ┆ case_creation_year ┆ case_creation_month ┆ insurance_company_one_hot │
│ ---   ┆ ---        ┆ ---     ┆ ---          ┆   ┆ ---             ┆ ---                ┆ ---                 ┆ ---                       │
│ u32   ┆ i64        ┆ str     ┆ str          ┆   ┆ f32             ┆ i32                ┆ str                 ┆ list[i8]                  │
╞═══════╪════════════╪═════════╪══════════════╪═══╪═════════════════╪════════════════════╪═════════════════════╪═══════════════════════════╡
│ 20742 ┆ 215663     ┆ Kjøkken ┆ kjøkken      ┆ … ┆ 3.1             ┆ 2023               ┆ 7                   ┆ [0, 0, … 0]               │
│ 11030 ┆ 254435     ┆ Stue    ┆ stue         ┆ … ┆ 31.4            ┆ 2022               ┆ 8                   ┆ [0, 0, … 0]               │
│ 15948 ┆ 168186     ┆ Gang    ┆ gang         ┆ … ┆ 4.1             ┆ 2024               ┆ 4                   ┆ [1, 0, … 0]               │
│ 6656  ┆ 210731     ┆ Kjøkken ┆ kjøkken      ┆ … ┆ 12.0            ┆ 2020               ┆ 3                   ┆ [0, 0, … 0]               │
│ 14470 ┆ 153731     ┆ 2 etg   ┆ ukjent       ┆ … ┆ 4.7             ┆ 2024               ┆ 7                   ┆ [0, 0, … 0]               │
└───────┴────────────┴─────────┴──────────────┴───┴─────────────────┴────────────────────┴─────────────────────┴───────────────────────────┘

--- is_hidden Distribution ---
/home/sve/cogito/recover-hackathon/eda.py:251: DeprecationWarning: `pl.count()` is deprecated. Please use `pl.len()` instead.
(Deprecated in version 0.20.5)
  pl.count().alias("count")
shape: (2, 2)
┌───────────┬────────┐
│ is_hidden ┆ count  │
│ ---       ┆ ---    │
│ bool      ┆ u32    │
╞═══════════╪════════╡
│ false     ┆ 230331 │
│ true      ┆ 51268  │
└───────────┴────────┘

--- Work Operation Distribution by is_hidden ---
Total visible operations: 230331
Total hidden operations: 51268

================================================================================
7. ENCODING VERIFICATION
================================================================================

--- Number of Clusters ---
Number of work operation clusters: 388

--- Room Clusters ---
Room to index mapping: {'andre områder': 0, 'kjøkken': 1, 'stue': 2, 'gang': 3, 'soverom': 4, 'bad': 5, 'bod': 6, 'vaskerom': 7, 'wc': 8, 'kjeller': 9, 'garasje': 10}

--- Verifying One-Hot Encoding ---
Sample codes: [1, 5, 10]
Encoded length: 388
Non-zero indices: [1, 5, 10]

--- Code to Name Mapping (first 10) ---
Code 0: Remont komplett dør
Code 1: Ny komplett dør
Code 2: Riv komplett dør
Code 3: Riv innv karmlist
Code 4: Ny innv karmlist
Code 5: Riv feielist
Code 6: Ny feielist
Code 7: Demont innv karmlist
Code 8: Demont komplett dør
Code 9: Riv innvendig dørterskel

================================================================================
8. DATA SPLIT CONSISTENCY
================================================================================

Train projects: 51009
Val projects: 3000
Test projects: 5994

Overlap train-val: 0
Overlap train-test: 0
Overlap val-test: 0

================================================================================
9. SUMMARY AND KEY INSIGHTS
================================================================================

KEY DATASET FACTS:
------------------
✓ Dataset Structure:
  - Training rooms: 27425
  - Validation rooms: 10827
  - Test rooms (for submission): 18299

✓ Work Operations:
  - Total unique operation clusters: 368
  - Cluster codes range: 0 to 387
  - Operations are encoded as one-hot vectors of length 388

✓ Room Information:
  - 11 standardized room types: ['andre områder', 'kjøkken', 'stue', 'gang', 'soverom', 'bad', 'bod', 'vaskerom', 'wc', 'kjeller', 'garasje']
  - Average operations per room: 10.28
  - Average rooms per project: 3.64

✓ Projects:
  - Training projects: 51009
  - Validation projects: 3000
  - Test projects: 5994
  - Project splits are non-overlapping: True

✓ Metadata Features:
  - Insurance companies: 14
  - Years: 2015 - 2025
  - Office distance range: 0.00 - 8474.00 km

✓ Data Format:
  - Each room has X (visible operations) and Y (hidden operations to predict)
  - X and Y are one-hot encoded vectors
  - Context from other rooms in same project available via 'calculus' field
  - Sampling strategy controls how many operations are hidden (default: 30-50%)

✓ Test Set:
  - Test set has all operations visible in X
  - Y is what needs to be predicted for submission
  - Must predict for 18299 unique room IDs


================================================================================
EDA COMPLETE!
================================================================================

Recommended next steps:
1. Visualize work operation co-occurrences
2. Analyze room-specific operation patterns
3. Examine temporal trends (if any)
4. Feature engineering based on metadata
5. Build baseline models to understand task difficulty
sve@sve-ThinkPad-T480:~/cogito/recover-hackathon$
"""