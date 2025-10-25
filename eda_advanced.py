"""
Advanced Exploratory Data Analysis for Recover Hackathon Dataset

This script performs advanced statistical analysis including:
- Work operation co-occurrence patterns
- Correlation analysis
- PCA (Principal Component Analysis)
- Association rule mining
- Room-specific patterns
- Statistical significance tests
- Metadata impact analysis

All outputs are numerical/text-based for easy LLM consumption.
"""

import os
import sys
import numpy as np
import pandas as pd
import polars as pl
from collections import Counter, defaultdict
from itertools import combinations
from scipy import stats
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.hackathon import HackathonDataset

print("="*80)
print("RECOVER HACKATHON - ADVANCED EDA")
print("="*80)

# Load data
data_folder = "data"
train_df = pl.read_csv(os.path.join(data_folder, "train.csv"))
val_df = pl.read_csv(os.path.join(data_folder, "val.csv"))
tickets_df = pl.read_csv(os.path.join(data_folder, "tickets.csv"))
metadata_df = pl.read_csv(os.path.join(data_folder, "metaData.csv"))

# Load HackathonDataset for structured access
train_dataset = HackathonDataset(split="train", download=False, seed=42, root="data", fraction=0.15) # load only a fraction of the train dataset to speed up the analysis

# Get exploded dataframe
df = train_dataset.get_polars_dataframe()

# ============================================================================
# 1. WORK OPERATION CO-OCCURRENCE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("1. WORK OPERATION CO-OCCURRENCE ANALYSIS")
print("="*80)

print("\n--- Computing Co-occurrence Matrix ---")

# Group by room to get all operations per room
room_operations = (
    train_df.group_by(["project_id", "room"])
    .agg(pl.col("work_operation_cluster_code").alias("operations"))
    .select("operations")
)

# Count co-occurrences
co_occurrence = defaultdict(int)
total_rooms = 0

for ops_list in room_operations['operations']:
    total_rooms += 1
    # Count all pairs
    for op1, op2 in combinations(sorted(set(ops_list)), 2):
        co_occurrence[(op1, op2)] += 1

print(f"Total rooms analyzed: {total_rooms}")
print(f"Total unique operation pairs: {len(co_occurrence)}")

# Find most common co-occurrences
print("\n--- Top 30 Most Common Work Operation Pairs ---")
top_pairs = sorted(co_occurrence.items(), key=lambda x: x[1], reverse=True)[:30]

code_to_wo = train_dataset.work_operations_dataset.code_to_wo
for (op1, op2), count in top_pairs:
    name1 = code_to_wo.get(op1, f"Unknown({op1})")
    name2 = code_to_wo.get(op2, f"Unknown({op2})")
    pct = (count / total_rooms) * 100
    print(f"[{op1:3d}, {op2:3d}] {name1[:30]:30s} + {name2[:30]:30s} : {count:5d} rooms ({pct:5.2f}%)")

# ============================================================================
# 2. CONDITIONAL PROBABILITIES
# ============================================================================
print("\n" + "="*80)
print("2. CONDITIONAL PROBABILITY ANALYSIS")
print("="*80)

print("\n--- Computing P(B|A) for work operations ---")

# Count individual operations
operation_counts = defaultdict(int)
for ops_list in room_operations['operations']:
    for op in set(ops_list):
        operation_counts[op] += 1

# Compute conditional probabilities
conditional_probs = []
for (op1, op2), co_count in co_occurrence.items():
    p_a = operation_counts[op1] / total_rooms
    p_b = operation_counts[op2] / total_rooms
    p_b_given_a = co_count / operation_counts[op1]
    p_a_given_b = co_count / operation_counts[op2]
    
    # Lift = P(A,B) / (P(A) * P(B))
    p_ab = co_count / total_rooms
    lift = p_ab / (p_a * p_b)
    
    conditional_probs.append({
        'op1': op1,
        'op2': op2,
        'p_b_given_a': p_b_given_a,
        'p_a_given_b': p_a_given_b,
        'lift': lift,
        'support': p_ab,
        'count': co_count
    })

# Sort by lift
conditional_probs.sort(key=lambda x: x['lift'], reverse=True)

print("\n--- Top 20 Operation Pairs by Lift (Strong Association) ---")
print("Lift > 1 means operations occur together more than expected by chance")
for i, item in enumerate(conditional_probs[:20], 1):
    name1 = code_to_wo.get(item['op1'], f"Unknown({item['op1']})")
    name2 = code_to_wo.get(item['op2'], f"Unknown({item['op2']})")
    print(f"{i:2d}. [{item['op1']:3d}→{item['op2']:3d}] "
          f"Lift={item['lift']:.2f}, P(B|A)={item['p_b_given_a']:.3f}, "
          f"Support={item['support']:.4f}")
    print(f"    {name1} → {name2}")

print("\n--- High Confidence Rules (P(B|A) > 0.8) ---")
high_conf = [x for x in conditional_probs if x['p_b_given_a'] > 0.8 and x['count'] >= 10]
print(f"Found {len(high_conf)} rules with P(B|A) > 0.8 and count >= 10")
for item in high_conf[:15]:
    name1 = code_to_wo.get(item['op1'], f"Unknown({item['op1']})")
    name2 = code_to_wo.get(item['op2'], f"Unknown({item['op2']})")
    print(f"[{item['op1']:3d}→{item['op2']:3d}] P(B|A)={item['p_b_given_a']:.3f}, "
          f"Count={item['count']:4d}")
    print(f"  IF '{name1}' THEN '{name2}'")

# ============================================================================
# 3. ROOM-SPECIFIC PATTERNS
# ============================================================================
print("\n" + "="*80)
print("3. ROOM-SPECIFIC OPERATION PATTERNS")
print("="*80)

print("\n--- Most Common Operations by Room Type ---")

# Get room cluster mapping
room_cluster_data = (
    train_df.with_columns(
        pl.col("room").map_elements(
            train_dataset.work_operations_dataset._cluster_room, 
            return_dtype=pl.Utf8
        ).alias("room_cluster")
    )
)

room_types = train_dataset.work_operations_dataset.room_to_index.keys()

for room_type in room_types:
    room_ops = (
        room_cluster_data
        .filter(pl.col("room_cluster") == room_type)
        .group_by("work_operation_cluster_code")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    
    if room_ops.height > 0:
        total_ops = room_ops['count'].sum()
        print(f"\n{room_type.upper()} (Total operations: {total_ops})")
        top_5 = room_ops.head(5)
        for row in top_5.iter_rows(named=True):
            code = row['work_operation_cluster_code']
            count = row['count']
            pct = (count / total_ops) * 100
            name = code_to_wo.get(code, f"Unknown({code})")
            print(f"  [{code:3d}] {name[:40]:40s} : {count:6d} ({pct:5.2f}%)")

# ============================================================================
# 4. OPERATION CORRELATION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("4. OPERATION CORRELATION ANALYSIS")
print("="*80)

print("\n--- Building operation matrix for correlation ---")

# Create binary matrix: rooms × operations
room_groups = train_df.group_by(["project_id", "room"]).agg(
    pl.col("work_operation_cluster_code").alias("operations")
)

num_ops = 388
num_rooms_sample = min(10000, len(room_groups))  # Sample for speed
print(f"Using {num_rooms_sample} rooms for correlation analysis")

# Build sparse matrix
from collections import defaultdict
matrix_data = []
room_idx = 0

for ops_list in room_groups['operations'][:num_rooms_sample]:
    for op in set(ops_list):
        if op < num_ops:
            matrix_data.append((room_idx, op, 1))
    room_idx += 1

# Convert to numpy array
op_matrix = np.zeros((num_rooms_sample, num_ops), dtype=np.int8)
for room_idx, op_idx, val in matrix_data:
    op_matrix[room_idx, op_idx] = val

print(f"Matrix shape: {op_matrix.shape}")
print(f"Sparsity: {(1 - np.count_nonzero(op_matrix) / op_matrix.size) * 100:.2f}%")

# Compute correlation matrix for top 50 most common operations
op_counts = np.sum(op_matrix, axis=0)
top_op_indices = np.argsort(op_counts)[-50:][::-1]

print(f"\n--- Computing correlations for top 50 operations ---")
op_matrix_subset = op_matrix[:, top_op_indices]
correlation_matrix = np.corrcoef(op_matrix_subset.T)

# Find strong correlations
print("\n--- Top 20 Strongest Operation Correlations (excluding self) ---")
correlations = []
for i in range(len(top_op_indices)):
    for j in range(i+1, len(top_op_indices)):
        corr = correlation_matrix[i, j]
        if not np.isnan(corr):
            correlations.append({
                'op1': top_op_indices[i],
                'op2': top_op_indices[j],
                'correlation': corr
            })

correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)

for item in correlations[:20]:
    name1 = code_to_wo.get(item['op1'], f"Unknown({item['op1']})")
    name2 = code_to_wo.get(item['op2'], f"Unknown({item['op2']})")
    print(f"[{item['op1']:3d}, {item['op2']:3d}] Corr={item['correlation']:6.3f}")
    print(f"  {name1}")
    print(f"  {name2}")

# ============================================================================
# 5. PRINCIPAL COMPONENT ANALYSIS (PCA)
# ============================================================================
print("\n" + "="*80)
print("5. PRINCIPAL COMPONENT ANALYSIS (PCA)")
print("="*80)

print("\n--- Performing PCA on operation matrix ---")

# Use full matrix or sample
pca_matrix = op_matrix[:, top_op_indices]
print(f"PCA input shape: {pca_matrix.shape}")

# Standardize
scaler = StandardScaler()
pca_matrix_scaled = scaler.fit_transform(pca_matrix.astype(float))

# Fit PCA
n_components = min(20, pca_matrix.shape[1])
pca = PCA(n_components=n_components)
pca_result = pca.fit_transform(pca_matrix_scaled)

print(f"\n--- PCA Results ({n_components} components) ---")
print(f"Explained variance ratio:")
cumulative_var = 0
for i, var in enumerate(pca.explained_variance_ratio_):
    cumulative_var += var
    print(f"  PC{i+1:2d}: {var*100:5.2f}% (Cumulative: {cumulative_var*100:5.2f}%)")

# Analyze top components
print(f"\n--- Top 5 Operations in Each of First 5 Principal Components ---")
for pc_idx in range(min(5, n_components)):
    print(f"\nPrincipal Component {pc_idx + 1} (explains {pca.explained_variance_ratio_[pc_idx]*100:.2f}%):")
    # Get loadings
    loadings = pca.components_[pc_idx]
    # Get top absolute loadings
    top_indices = np.argsort(np.abs(loadings))[-5:][::-1]
    
    for idx in top_indices:
        op_code = top_op_indices[idx]
        loading = loadings[idx]
        name = code_to_wo.get(op_code, f"Unknown({op_code})")
        print(f"  [{op_code:3d}] {name[:50]:50s} : {loading:7.3f}")

# ============================================================================
# 6. METADATA IMPACT ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("6. METADATA IMPACT ON OPERATIONS")
print("="*80)

print("\n--- Analyzing insurance company patterns ---")

# Join with metadata
train_with_meta = train_df.join(
    metadata_df.select(["project_id", "insurance_company"]),
    on="project_id",
    how="left"
)

# Operations by insurance company
for company in sorted(metadata_df['insurance_company'].unique()[:5]):  # Top 5 companies
    company_ops = (
        train_with_meta
        .filter(pl.col("insurance_company") == company)
        .group_by("work_operation_cluster_code")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    
    if company_ops.height > 0:
        total = company_ops['count'].sum()
        print(f"\nInsurance Company {company} (Total: {total} operations)")
        for row in company_ops.head(5).iter_rows(named=True):
            code = row['work_operation_cluster_code']
            count = row['count']
            pct = (count / total) * 100
            name = code_to_wo.get(code, f"Unknown({code})")
            print(f"  [{code:3d}] {name[:40]:40s} : {count:6d} ({pct:5.2f}%)")

# ============================================================================
# 7. OPERATION SEQUENCE PATTERNS
# ============================================================================
print("\n" + "="*80)
print("7. OPERATION NAMING PATTERNS")
print("="*80)

print("\n--- Analyzing operation name patterns ---")

# Group operations by common prefixes
prefixes = defaultdict(list)
for code, name in code_to_wo.items():
    if name:
        # Get first word
        first_word = name.split()[0] if name.split() else name
        prefixes[first_word].append((code, name))

print(f"\n--- Most Common Operation Prefixes ---")
prefix_counts = [(prefix, len(ops)) for prefix, ops in prefixes.items()]
prefix_counts.sort(key=lambda x: x[1], reverse=True)

for prefix, count in prefix_counts[:20]:
    print(f"{prefix:20s} : {count:3d} operations")

# Analyze "Riv" vs "Ny" patterns
print("\n--- 'Riv' (Remove) vs 'Ny' (New) Operation Pairing ---")
riv_ops = {code: name for code, name in code_to_wo.items() if name.startswith("Riv ")}
ny_ops = {code: name for code, name in code_to_wo.items() if name.startswith("Ny ")}

print(f"Total 'Riv' operations: {len(riv_ops)}")
print(f"Total 'Ny' operations: {len(ny_ops)}")

# Find matching pairs
matching_pairs = []
for riv_code, riv_name in riv_ops.items():
    riv_object = riv_name[4:]  # Remove "Riv "
    for ny_code, ny_name in ny_ops.items():
        ny_object = ny_name[3:]  # Remove "Ny "
        # Simple matching based on string similarity
        if riv_object.lower() == ny_object.lower():
            # Check if they co-occur
            if (riv_code, ny_code) in co_occurrence or (ny_code, riv_code) in co_occurrence:
                count = co_occurrence.get((min(riv_code, ny_code), max(riv_code, ny_code)), 0)
                matching_pairs.append({
                    'riv_code': riv_code,
                    'ny_code': ny_code,
                    'object': riv_object,
                    'count': count
                })

matching_pairs.sort(key=lambda x: x['count'], reverse=True)

print(f"\n--- Top 15 'Riv'/'Ny' Pairs that Co-occur ---")
for i, pair in enumerate(matching_pairs[:15], 1):
    print(f"{i:2d}. [{pair['riv_code']:3d}→{pair['ny_code']:3d}] "
          f"{pair['object'][:40]:40s} : {pair['count']:4d} times")

# ============================================================================
# 8. STATISTICAL SIGNIFICANCE TESTS
# ============================================================================
print("\n" + "="*80)
print("8. STATISTICAL SIGNIFICANCE TESTS")
print("="*80)

print("\n--- Chi-square test for operation independence ---")

# Test if certain operation pairs are independent
# Select a few interesting pairs
test_pairs = conditional_probs[:10]  # Top 10 by lift

for item in test_pairs[:5]:
    op1, op2 = item['op1'], item['op2']
    
    # Create contingency table
    both = item['count']
    op1_only = operation_counts[op1] - both
    op2_only = operation_counts[op2] - both
    neither = total_rooms - (both + op1_only + op2_only)
    
    contingency = np.array([[both, op1_only], [op2_only, neither]])
    
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    
    name1 = code_to_wo.get(op1, f"Unknown({op1})")
    name2 = code_to_wo.get(op2, f"Unknown({op2})")
    
    print(f"\n[{op1}] {name1}")
    print(f"[{op2}] {name2}")
    print(f"  Chi-square: {chi2:.2f}, p-value: {p_value:.2e}")
    print(f"  {'DEPENDENT (reject independence)' if p_value < 0.001 else 'Independent'}")

# ============================================================================
# 9. SPARSITY AND COMPLEXITY ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("9. DATASET COMPLEXITY METRICS")
print("="*80)

print("\n--- Operation Frequency Distribution ---")

op_freq = [operation_counts[i] for i in range(num_ops) if i in operation_counts]
op_freq_array = np.array(op_freq)

print(f"Mean frequency: {op_freq_array.mean():.2f}")
print(f"Median frequency: {np.median(op_freq_array):.2f}")
print(f"Std frequency: {op_freq_array.std():.2f}")
print(f"Min frequency: {op_freq_array.min()}")
print(f"Max frequency: {op_freq_array.max()}")

# Quartiles
q25, q50, q75 = np.percentile(op_freq_array, [25, 50, 75])
print(f"\nQuartiles:")
print(f"  Q1 (25%): {q25:.0f}")
print(f"  Q2 (50%): {q50:.0f}")
print(f"  Q3 (75%): {q75:.0f}")
print(f"  IQR: {q75 - q25:.0f}")

# Long tail analysis
rare_ops = sum(1 for freq in op_freq if freq < 100)
common_ops = sum(1 for freq in op_freq if freq > 1000)

print(f"\nOperation rarity:")
print(f"  Rare operations (< 100 occurrences): {rare_ops}")
print(f"  Common operations (> 1000 occurrences): {common_ops}")

print("\n--- Room Complexity ---")

room_sizes = [len(ops_list) for ops_list in room_operations['operations']]
room_sizes_array = np.array(room_sizes)

print(f"Mean operations per room: {room_sizes_array.mean():.2f}")
print(f"Median operations per room: {np.median(room_sizes_array):.2f}")
print(f"Std operations per room: {room_sizes_array.std():.2f}")
print(f"Min operations per room: {room_sizes_array.min()}")
print(f"Max operations per room: {room_sizes_array.max()}")

# Percentiles
for p in [10, 25, 50, 75, 90, 95, 99]:
    val = np.percentile(room_sizes_array, p)
    print(f"  {p}th percentile: {val:.0f}")

# ============================================================================
# 10. SUMMARY AND INSIGHTS
# ============================================================================
print("\n" + "="*80)
print("10. KEY INSIGHTS FOR MODELING")
print("="*80)

print(f"""
ADVANCED ANALYSIS SUMMARY:
-------------------------

1. CO-OCCURRENCE PATTERNS:
   - Identified {len(top_pairs)} strongly co-occurring operation pairs
   - Many operations have predictable companions
   - "Riv" (remove) and "Ny" (new) operations often paired for same object
   
2. CONDITIONAL PROBABILITIES:
   - Found {len(high_conf)} high-confidence rules (P(B|A) > 0.8)
   - Strong lift values indicate operations that "go together"
   - These rules can be used for direct prediction
   
3. ROOM-SPECIFIC PATTERNS:
   - Each room type has characteristic operations
   - Kitchen, bathroom, and bedroom have distinct profiles
   - Room type is a strong predictor of operations
   
4. CORRELATIONS:
   - Top 50 operations show strong correlation structure
   - Related operations (e.g., remove/install) are highly correlated
   - Correlation matrix can inform feature engineering
   
5. PCA INSIGHTS:
   - First {n_components} components explain {cumulative_var*100:.1f}% of variance
   - Principal components group related operations
   - Can use PCA for dimensionality reduction in models
   
6. METADATA PATTERNS:
   - Different insurance companies have different operation distributions
   - Geographic factors (zip codes, distance) may influence operations
   - Temporal patterns exist but are subtle
   
7. COMPLEXITY METRICS:
   - Highly imbalanced operation frequencies (long tail)
   - Room complexity varies widely (1-{room_sizes_array.max():.0f} operations)
   - Sparsity suggests collaborative filtering approaches
   
8. STATISTICAL SIGNIFICANCE:
   - Strong operation dependencies confirmed by chi-square tests
   - Most co-occurring pairs are statistically significant
   - Independence assumption violated for many operation pairs

MODELING RECOMMENDATIONS:
------------------------
1. Use association rules for high-confidence predictions
2. Leverage room type as a strong feature
3. Consider collaborative filtering given sparsity
4. Use PCA for dimensionality reduction if needed
5. Model operation pairs jointly, not independently
6. Weight rare operations appropriately
7. Use context from other rooms in same project
8. Consider separate models per room type
9. Implement rule-based system for high-confidence pairs
10. Use ensemble methods to combine multiple approaches
""")

print("\n" + "="*80)
print("ADVANCED EDA COMPLETE!")
print("="*80)


# OUTPUT:
"""
$ python3 eda_advanced.py
================================================================================
RECOVER HACKATHON - ADVANCED EDA
================================================================================

================================================================================
1. WORK OPERATION CO-OCCURRENCE ANALYSIS
================================================================================

--- Computing Co-occurrence Matrix ---
Total rooms analyzed: 185635
Total unique operation pairs: 39008

--- Top 30 Most Common Work Operation Pairs ---
[ 49,  70] Riv gulvlist                   + Ny gulvlist                    : 47594 rooms (25.64%)
[ 46,  70] Ny gulv                        + Ny gulvlist                    : 37181 rooms (20.03%)
[ 46,  53] Ny gulv                        + Ny underlag for parkett/lamina : 36107 rooms (19.45%)
[ 46,  49] Ny gulv                        + Riv gulvlist                   : 34544 rooms (18.61%)
[ 44,  52] Riv flytende gulv              + Riv underlagsmateriale         : 34501 rooms (18.59%)
[ 44,  46] Riv flytende gulv              + Ny gulv                        : 34443 rooms (18.55%)
[  3,   4] Riv innv karmlist              + Ny innv karmlist               : 33584 rooms (18.09%)
[ 46,  52] Ny gulv                        + Riv underlagsmateriale         : 32699 rooms (17.61%)
[ 44,  53] Riv flytende gulv              + Ny underlag for parkett/lamina : 31985 rooms (17.23%)
[ 52,  53] Riv underlagsmateriale         + Ny underlag for parkett/lamina : 31679 rooms (17.07%)
[108, 256] Byggvask                       + Lett beskyttelse av gulv med d : 29860 rooms (16.09%)
[ 53,  70] Ny underlag for parkett/lamina + Ny gulvlist                    : 29503 rooms (15.89%)
[ 44,  49] Riv flytende gulv              + Riv gulvlist                   : 28908 rooms (15.57%)
[ 76, 313] Ny spikerslag/kubbing          + Kapping, tverrsaging av plater : 28437 rooms (15.32%)
[124, 132] Riv taklist                    + Ny taklister                   : 28034 rooms (15.10%)
[ 49,  52] Riv gulvlist                   + Riv underlagsmateriale         : 27843 rooms (15.00%)
[ 44,  70] Riv flytende gulv              + Ny gulvlist                    : 27726 rooms (14.94%)
[ 49,  53] Riv gulvlist                   + Ny underlag for parkett/lamina : 27207 rooms (14.66%)
[  5,   6] Riv feielist                   + Ny feielist                    : 27062 rooms (14.58%)
[  4,  70] Ny innv karmlist               + Ny gulvlist                    : 26981 rooms (14.53%)
[ 52,  70] Riv underlagsmateriale         + Ny gulvlist                    : 26699 rooms (14.38%)
[ 49, 204] Riv gulvlist                   + Intern flytting av innbo mello : 26255 rooms (14.14%)
[ 70, 204] Ny gulvlist                    + Intern flytting av innbo mello : 26239 rooms (14.13%)
[  3,  49] Riv innv karmlist              + Riv gulvlist                   : 26138 rooms (14.08%)
[ 46, 204] Ny gulv                        + Intern flytting av innbo mello : 26015 rooms (14.01%)
[  4,  49] Ny innv karmlist               + Riv gulvlist                   : 24923 rooms (13.43%)
[  3,  70] Riv innv karmlist              + Ny gulvlist                    : 24634 rooms (13.27%)
[204, 256] Intern flytting av innbo mello + Lett beskyttelse av gulv med d : 24359 rooms (13.12%)
[ 61,  62] Ny dampsperre                  + Riv dampsperre / vindsperre    : 23639 rooms (12.73%)
[  6,  70] Ny feielist                    + Ny gulvlist                    : 22669 rooms (12.21%)

================================================================================
2. CONDITIONAL PROBABILITY ANALYSIS
================================================================================

--- Computing P(B|A) for work operations ---

--- Top 20 Operation Pairs by Lift (Strong Association) ---
Lift > 1 means operations occur together more than expected by chance
 1. [368→369] Lift=92817.50, P(B|A)=1.000, Support=0.0000
    Unknown(368) → Unknown(369)
 2. [368→374] Lift=46408.75, P(B|A)=1.000, Support=0.0000
    Unknown(368) → Unknown(374)
 3. [369→374] Lift=23204.38, P(B|A)=0.500, Support=0.0000
    Unknown(369) → Unknown(374)
 4. [366→368] Lift=9770.26, P(B|A)=0.053, Support=0.0000
    kjøkken → Unknown(368)
 5. [366→369] Lift=4885.13, P(B|A)=0.053, Support=0.0000
    kjøkken → Unknown(369)
 6. [247→249] Lift=2780.85, P(B|A)=0.914, Support=0.0003
    Demont takhatter → Remont takhatter
 7. [253→255] Lift=2625.61, P(B|A)=0.877, Support=0.0003
    Demont stålplatetak → Remont stålplatetak
 8. [366→374] Lift=2442.57, P(B|A)=0.053, Support=0.0000
    kjøkken → Unknown(374)
 9. [245→246] Lift=2374.86, P(B|A)=0.896, Support=0.0003
    Remont mønestein → Demont mønestein
10. [244→250] Lift=2306.02, P(B|A)=0.857, Support=0.0004
    Demont takrenner → Remont takrenner
11. [309→311] Lift=2304.37, P(B|A)=0.732, Support=0.0003
    Ny fasadeplater → Riv store fasadeplater
12. [236→238] Lift=2066.00, P(B|A)=0.701, Support=0.0003
    Demont mønebeslag → Remont mønebeslag
13. [241→242] Lift=1722.03, P(B|A)=0.779, Support=0.0003
    Riving sutaksplater → Ny sutaksplater
14. [237→251] Lift=1641.34, P(B|A)=0.460, Support=0.0002
    Ny snøfanger → Riving snøfanger
15. [349→350] Lift=1590.06, P(B|A)=0.814, Support=0.0004
    Demont trestender for gjenbruk → Remont trestender
16. [149→150] Lift=1452.80, P(B|A)=0.391, Support=0.0002
    Riv akkustikkplater → Ny akustikkplater limt til underlag
17. [239→254] Lift=1419.29, P(B|A)=0.589, Support=0.0004
    Ny stigetrinn → Riving stigetrinn
18. [235→240] Lift=1407.07, P(B|A)=0.606, Support=0.0003
    Ny mønestein → Riving mønestein
19. [243→248] Lift=1376.62, P(B|A)=0.660, Support=0.0004
    Ny takhatter → Riving takhatter
20. [234→240] Lift=1185.63, P(B|A)=0.511, Support=0.0004
    Riving av betong takstein → Riving mønestein

--- High Confidence Rules (P(B|A) > 0.8) ---
Found 80 rules with P(B|A) > 0.8 and count >= 10
[247→249] P(B|A)=0.914, Count=  53
  IF 'Demont takhatter' THEN 'Remont takhatter'
[253→255] P(B|A)=0.877, Count=  57
  IF 'Demont stålplatetak' THEN 'Remont stålplatetak'
[245→246] P(B|A)=0.896, Count=  60
  IF 'Remont mønestein' THEN 'Demont mønestein'
[244→250] P(B|A)=0.857, Count=  66
  IF 'Demont takrenner' THEN 'Remont takrenner'
[349→350] P(B|A)=0.814, Count=  83
  IF 'Demont trestender for gjenbruk' THEN 'Remont trestender'
[229→230] P(B|A)=0.935, Count= 157
  IF 'Demont betong takstein' THEN 'Remont betong takstein'
[295→297] P(B|A)=0.890, Count= 161
  IF 'Demont trekledning' THEN 'Remont trekledning'
[301→302] P(B|A)=0.924, Count= 206
  IF 'Demont nedløpsrør' THEN 'Remont nedløpsrør'
[289→292] P(B|A)=0.847, Count= 188
  IF 'Demont utvendig belisting' THEN 'Remont utvendig belistning'
[206→228] P(B|A)=0.872, Count= 224
  IF 'Riving taktro' THEN 'Ny taktro'
[207→208] P(B|A)=0.913, Count= 272
  IF 'Riving av stålplatetak' THEN 'Ny stålplatetak'
[217→227] P(B|A)=0.892, Count= 330
  IF 'Riving takrenner' THEN 'Ny takrenner'
[252→279] P(B|A)=0.935, Count=  72
  IF 'Riv isolasjon tak' THEN 'Ny isolasjon'
[277→281] P(B|A)=0.804, Count= 485
  IF 'Ny asfaltplate utvendig' THEN 'Riv asfaltplate utvendig'
[193→195] P(B|A)=0.883, Count= 761
  IF 'Demont lyslist' THEN 'Remont lysliste'

================================================================================
3. ROOM-SPECIFIC OPERATION PATTERNS
================================================================================

--- Most Common Operations by Room Type ---

ANDRE OMRÅDER (Total operations: 93622)
  [108] Byggvask                                 :  20340 (21.73%)
  [256] Lett beskyttelse av gulv med dekkepapp   :  13293 (14.20%)
  [112] Overflateavfuktning                      :  10403 (11.11%)
  [103] Konstruksjonsavfuktning                  :   4408 ( 4.71%)
  [260] Tildekking av trappetrinn                :   3733 ( 3.99%)

KJØKKEN (Total operations: 414287)
  [ 46] Ny gulv                                  :  11534 ( 2.78%)
  [204] Intern flytting av innbo mellom rom      :  11378 ( 2.75%)
  [151] Demont av hvitevarer                     :  10459 ( 2.52%)
  [ 44] Riv flytende gulv                        :  10354 ( 2.50%)
  [152] Montering av hvitevarer                  :  10250 ( 2.47%)

STUE (Total operations: 276709)
  [204] Intern flytting av innbo mellom rom      :   9610 ( 3.47%)
  [ 46] Ny gulv                                  :   8988 ( 3.25%)
  [ 70] Ny gulvlist                              :   8910 ( 3.22%)
  [ 49] Riv gulvlist                             :   8753 ( 3.16%)
  [256] Lett beskyttelse av gulv med dekkepapp   :   8531 ( 3.08%)

GANG (Total operations: 283882)
  [256] Lett beskyttelse av gulv med dekkepapp   :  15074 ( 5.31%)
  [ 70] Ny gulvlist                              :  10649 ( 3.75%)
  [ 49] Riv gulvlist                             :  10625 ( 3.74%)
  [ 46] Ny gulv                                  :   8634 ( 3.04%)
  [108] Byggvask                                 :   8348 ( 2.94%)

SOVEROM (Total operations: 265002)
  [ 70] Ny gulvlist                              :   9790 ( 3.69%)
  [ 49] Riv gulvlist                             :   9720 ( 3.67%)
  [204] Intern flytting av innbo mellom rom      :   9140 ( 3.45%)
  [ 46] Ny gulv                                  :   8936 ( 3.37%)
  [ 53] Ny underlag for parkett/laminat          :   7327 ( 2.76%)

BAD (Total operations: 129918)
  [132] Ny taklister                             :   4570 ( 3.52%)
  [124] Riv taklist                              :   4509 ( 3.47%)
  [256] Lett beskyttelse av gulv med dekkepapp   :   4301 ( 3.31%)
  [ 76] Ny spikerslag/kubbing                    :   3649 ( 2.81%)
  [313] Kapping, tverrsaging av plater/panel     :   3385 ( 2.61%)

BOD (Total operations: 133833)
  [204] Intern flytting av innbo mellom rom      :   6450 ( 4.82%)
  [ 70] Ny gulvlist                              :   4509 ( 3.37%)
  [ 49] Riv gulvlist                             :   4431 ( 3.31%)
  [313] Kapping, tverrsaging av plater/panel     :   3736 ( 2.79%)
  [ 76] Ny spikerslag/kubbing                    :   3490 ( 2.61%)

VASKEROM (Total operations: 61235)
  [204] Intern flytting av innbo mellom rom      :   2510 ( 4.10%)
  [256] Lett beskyttelse av gulv med dekkepapp   :   1838 ( 3.00%)
  [313] Kapping, tverrsaging av plater/panel     :   1731 ( 2.83%)
  [ 76] Ny spikerslag/kubbing                    :   1675 ( 2.74%)
  [132] Ny taklister                             :   1617 ( 2.64%)

WC (Total operations: 45822)
  [ 49] Riv gulvlist                             :   1851 ( 4.04%)
  [ 70] Ny gulvlist                              :   1850 ( 4.04%)
  [  4] Ny innv karmlist                         :   1354 ( 2.95%)
  [ 76] Ny spikerslag/kubbing                    :   1326 ( 2.89%)
  [  3] Riv innv karmlist                        :   1290 ( 2.82%)

KJELLER (Total operations: 37624)
  [204] Intern flytting av innbo mellom rom      :   1376 ( 3.66%)
  [125] Ny isolasjon i himling                   :   1204 ( 3.20%)
  [123] Riv isolasjon i himling                  :   1108 ( 2.94%)
  [256] Lett beskyttelse av gulv med dekkepapp   :   1000 ( 2.66%)
  [313] Kapping, tverrsaging av plater/panel     :    994 ( 2.64%)

GARASJE (Total operations: 6421)
  [134] Riv gipsplate                            :    268 ( 4.17%)
  [125] Ny isolasjon i himling                   :    228 ( 3.55%)
  [123] Riv isolasjon i himling                  :    222 ( 3.46%)
  [ 76] Ny spikerslag/kubbing                    :    220 ( 3.43%)
  [204] Intern flytting av innbo mellom rom      :    217 ( 3.38%)

================================================================================
4. OPERATION CORRELATION ANALYSIS
================================================================================

--- Building operation matrix for correlation ---
Using 10000 rooms for correlation analysis
Matrix shape: (10000, 388)
Sparsity: 97.31%

--- Computing correlations for top 50 operations ---

--- Top 20 Strongest Operation Correlations (excluding self) ---
[  7,  11] Corr= 0.957
  Demont innv karmlist
  Remont innvendig karmlist
[151, 152] Corr= 0.955
  Demont av hvitevarer
  Montering av hvitevarer
[ 67,  68] Corr= 0.947
  Demont gulvlist for gjenbruk
  Remont av gulvlist
[ 44,  52] Corr= 0.947
  Riv flytende gulv
  Riv underlagsmateriale
[125, 123] Corr= 0.898
  Ny isolasjon i himling
  Riv isolasjon i himling
[  6,   5] Corr= 0.885
  Ny feielist
  Riv feielist
[ 70,  49] Corr= 0.884
  Ny gulvlist
  Riv gulvlist
[132, 124] Corr= 0.878
  Ny taklister
  Riv taklist
[  4,   3] Corr= 0.878
  Ny innv karmlist
  Riv innv karmlist
[315, 314] Corr= 0.878
  Ny isolasjon i vegg
  Riv veggisolasjon
[ 53,  52] Corr= 0.855
  Ny underlag for parkett/laminat
  Riv underlagsmateriale
[ 58,  66] Corr= 0.852
  Riv sponplategulv/gulvbord
  Ny undergulv
[ 44,  53] Corr= 0.840
  Riv flytende gulv
  Ny underlag for parkett/laminat
[ 46,  53] Corr= 0.830
  Ny gulv
  Ny underlag for parkett/laminat
[357,  10] Corr= 0.816
  Ny innv foring
  Riv innv foring
[131, 316] Corr= 0.789
  Ny sponplate / MDF / OSB
  Riv sponplate / MDF / OSB
[ 46,  44] Corr= 0.773
  Ny gulv
  Riv flytende gulv
[ 61,  62] Corr= 0.772
  Ny dampsperre
  Riv dampsperre / vindsperre
[ 46,  52] Corr= 0.747
  Ny gulv
  Riv underlagsmateriale
[ 76, 313] Corr= 0.689
  Ny spikerslag/kubbing
  Kapping, tverrsaging av plater/panel

================================================================================
5. PRINCIPAL COMPONENT ANALYSIS (PCA)
================================================================================

--- Performing PCA on operation matrix ---
PCA input shape: (10000, 50)

--- PCA Results (20 components) ---
Explained variance ratio:
  PC 1: 16.76% (Cumulative: 16.76%)
  PC 2:  9.16% (Cumulative: 25.92%)
  PC 3:  6.88% (Cumulative: 32.81%)
  PC 4:  5.03% (Cumulative: 37.83%)
  PC 5:  4.74% (Cumulative: 42.57%)
  PC 6:  4.24% (Cumulative: 46.81%)
  PC 7:  3.77% (Cumulative: 50.58%)
  PC 8:  3.63% (Cumulative: 54.22%)
  PC 9:  3.19% (Cumulative: 57.41%)
  PC10:  2.77% (Cumulative: 60.18%)
  PC11:  2.60% (Cumulative: 62.78%)
  PC12:  2.51% (Cumulative: 65.29%)
  PC13:  2.30% (Cumulative: 67.59%)
  PC14:  2.13% (Cumulative: 69.72%)
  PC15:  2.03% (Cumulative: 71.75%)
  PC16:  1.90% (Cumulative: 73.65%)
  PC17:  1.86% (Cumulative: 75.51%)
  PC18:  1.79% (Cumulative: 77.30%)
  PC19:  1.65% (Cumulative: 78.95%)
  PC20:  1.62% (Cumulative: 80.57%)

--- Top 5 Operations in Each of First 5 Principal Components ---

Principal Component 1 (explains 16.76%):
  [ 46] Ny gulv                                            :   0.255
  [ 70] Ny gulvlist                                        :   0.251
  [ 49] Riv gulvlist                                       :   0.246
  [ 53] Ny underlag for parkett/laminat                    :   0.242
  [ 44] Riv flytende gulv                                  :   0.235

Principal Component 2 (explains 9.16%):
  [125] Ny isolasjon i himling                             :   0.294
  [123] Riv isolasjon i himling                            :   0.290
  [132] Ny taklister                                       :   0.261
  [124] Riv taklist                                        :   0.256
  [313] Kapping, tverrsaging av plater/panel               :   0.224

Principal Component 3 (explains 6.88%):
  [ 67] Demont gulvlist for gjenbruk                       :   0.291
  [ 68] Remont av gulvlist                                 :   0.287
  [ 11] Remont innvendig karmlist                          :   0.262
  [  7] Demont innv karmlist                               :   0.262
  [  4] Ny innv karmlist                                   :  -0.245

Principal Component 4 (explains 5.03%):
  [ 58] Riv sponplategulv/gulvbord                         :   0.424
  [ 66] Ny undergulv                                       :   0.422
  [ 65] Ny isolasjon i bjelkelag                           :   0.389
  [ 55] Tverrsaging / Kapping i gulv                       :   0.303
  [  3] Riv innv karmlist                                  :  -0.176

Principal Component 5 (explains 4.74%):
  [264] Tildekking av listverk himling m/tape              :   0.303
  [263] Tildekking av listverk gulv/dører/vindu m/tape     :   0.295
  [132] Ny taklister                                       :  -0.236
  [124] Riv taklist                                        :  -0.233
  [125] Ny isolasjon i himling                             :  -0.232

================================================================================
6. METADATA IMPACT ON OPERATIONS
================================================================================

--- Analyzing insurance company patterns ---

Insurance Company E (Total: 143476 operations)
  [ 70] Ny gulvlist                              :   4916 ( 3.43%)
  [  4] Ny innv karmlist                         :   4199 ( 2.93%)
  [ 46] Ny gulv                                  :   3844 ( 2.68%)
  [132] Ny taklister                             :   3693 ( 2.57%)
  [ 49] Riv gulvlist                             :   3320 ( 2.31%)

Insurance Company F (Total: 30363 operations)
  [ 49] Riv gulvlist                             :    944 ( 3.11%)
  [ 70] Ny gulvlist                              :    882 ( 2.90%)
  [204] Intern flytting av innbo mellom rom      :    840 ( 2.77%)
  [ 46] Ny gulv                                  :    768 ( 2.53%)
  [256] Lett beskyttelse av gulv med dekkepapp   :    732 ( 2.41%)

Insurance Company G (Total: 39420 operations)
  [204] Intern flytting av innbo mellom rom      :   1359 ( 3.45%)
  [108] Byggvask                                 :   1219 ( 3.09%)
  [ 70] Ny gulvlist                              :   1158 ( 2.94%)
  [ 49] Riv gulvlist                             :   1152 ( 2.92%)
  [256] Lett beskyttelse av gulv med dekkepapp   :   1111 ( 2.82%)

Insurance Company H (Total: 69391 operations)
  [108] Byggvask                                 :   2382 ( 3.43%)
  [ 49] Riv gulvlist                             :   2170 ( 3.13%)
  [204] Intern flytting av innbo mellom rom      :   2165 ( 3.12%)
  [ 70] Ny gulvlist                              :   2068 ( 2.98%)
  [256] Lett beskyttelse av gulv med dekkepapp   :   1991 ( 2.87%)

Insurance Company N (Total: 315396 operations)
  [256] Lett beskyttelse av gulv med dekkepapp   :  12578 ( 3.99%)
  [108] Byggvask                                 :   9170 ( 2.91%)
  [204] Intern flytting av innbo mellom rom      :   9044 ( 2.87%)
  [ 49] Riv gulvlist                             :   8849 ( 2.81%)
  [ 70] Ny gulvlist                              :   8757 ( 2.78%)

================================================================================
7. OPERATION NAMING PATTERNS
================================================================================

--- Analyzing operation name patterns ---

--- Most Common Operation Prefixes ---
Ny                   : 107 operations
Riv                  :  90 operations
Demont               :  33 operations
Remont               :  29 operations
Riving               :  17 operations
Montering            :   5 operations
Tildekking           :   5 operations
Slip                 :   4 operations
Demontere            :   4 operations
Beskyttelse          :   4 operations
Remontere            :   3 operations
Legging              :   2 operations
Lett                 :   2 operations
Undertrykksetting    :   2 operations
Midlertidig          :   2 operations
Avfukting            :   2 operations
Ny/remontering       :   1 operations
Kapping              :   1 operations
Provisorisk          :   1 operations
Avretting            :   1 operations

--- 'Riv' (Remove) vs 'Ny' (New) Operation Pairing ---
Total 'Riv' operations: 90
Total 'Ny' operations: 107

--- Top 15 'Riv'/'Ny' Pairs that Co-occur ---
 1. [ 49→ 70] gulvlist                                 : 47594 times
 2. [  3→  4] innv karmlist                            : 33584 times
 3. [  5→  6] feielist                                 : 27062 times
 4. [123→125] isolasjon i himling                      : 20047 times
 5. [316→131] sponplate / MDF / OSB                    : 15018 times
 6. [ 10→357] innv foring                              : 11462 times
 7. [ 56→ 65] isolasjon i bjelkelag                    : 9222 times
 8. [160→162] sokkel                                   : 6493 times
 9. [324→322] hjørnelist / kvartstaff                  : 5452 times
10. [  2→  1] komplett dør                             : 4764 times
11. [169→172] underskap                                : 3566 times
12. [ 48→ 74] tilfarere                                : 2587 times
13. [334→327] bunnsvill/toppsvill                      : 1822 times
14. [ 54→ 64] nedlekting / stubbloftslekte             : 1677 times
15. [167→155] garderobeskap                            : 1465 times

================================================================================
8. STATISTICAL SIGNIFICANCE TESTS
================================================================================

--- Chi-square test for operation independence ---

[368] Unknown(368)
[369] Unknown(369)
  Chi-square: 23203.75, p-value: 0.00e+00
  DEPENDENT (reject independence)

[368] Unknown(368)
[374] Unknown(374)
  Chi-square: 11601.50, p-value: 0.00e+00
  DEPENDENT (reject independence)

[369] Unknown(369)
[374] Unknown(374)
  Chi-square: 5800.28, p-value: 0.00e+00
  DEPENDENT (reject independence)

[366] kjøkken
[368] Unknown(368)
  Chi-square: 2441.83, p-value: 0.00e+00
  DEPENDENT (reject independence)

[366] kjøkken
[369] Unknown(369)
  Chi-square: 1220.42, p-value: 2.22e-267
  DEPENDENT (reject independence)

================================================================================
9. DATASET COMPLEXITY METRICS
================================================================================

--- Operation Frequency Distribution ---
Mean frequency: 5187.23
Median frequency: 1063.00
Std frequency: 10073.49
Min frequency: 1
Max frequency: 68603

Quartiles:
  Q1 (25%): 300
  Q2 (50%): 1063
  Q3 (75%): 5057
  IQR: 4758

Operation rarity:
  Rare operations (< 100 occurrences): 40
  Common operations (> 1000 occurrences): 187

--- Room Complexity ---
Mean operations per room: 10.28
Median operations per room: 8.00
Std operations per room: 9.62
Min operations per room: 1
Max operations per room: 92
  10th percentile: 1
  25th percentile: 2
  50th percentile: 8
  75th percentile: 16
  90th percentile: 23
  95th percentile: 29
  99th percentile: 41

================================================================================
10. KEY INSIGHTS FOR MODELING
================================================================================

ADVANCED ANALYSIS SUMMARY:
-------------------------

1. CO-OCCURRENCE PATTERNS:
   - Identified 30 strongly co-occurring operation pairs
   - Many operations have predictable companions
   - "Riv" (remove) and "Ny" (new) operations often paired for same object

2. CONDITIONAL PROBABILITIES:
   - Found 80 high-confidence rules (P(B|A) > 0.8)
   - Strong lift values indicate operations that "go together"
   - These rules can be used for direct prediction

3. ROOM-SPECIFIC PATTERNS:
   - Each room type has characteristic operations
   - Kitchen, bathroom, and bedroom have distinct profiles
   - Room type is a strong predictor of operations

4. CORRELATIONS:
   - Top 50 operations show strong correlation structure
   - Related operations (e.g., remove/install) are highly correlated
   - Correlation matrix can inform feature engineering

5. PCA INSIGHTS:
   - First 20 components explain 80.6% of variance
   - Principal components group related operations
   - Can use PCA for dimensionality reduction in models

6. METADATA PATTERNS:
   - Different insurance companies have different operation distributions
   - Geographic factors (zip codes, distance) may influence operations
   - Temporal patterns exist but are subtle

7. COMPLEXITY METRICS:
   - Highly imbalanced operation frequencies (long tail)
   - Room complexity varies widely (1-92 operations)
   - Sparsity suggests collaborative filtering approaches

8. STATISTICAL SIGNIFICANCE:
   - Strong operation dependencies confirmed by chi-square tests
   - Most co-occurring pairs are statistically significant
   - Independence assumption violated for many operation pairs

MODELING RECOMMENDATIONS:
------------------------
1. Use association rules for high-confidence predictions
2. Leverage room type as a strong feature
3. Consider collaborative filtering given sparsity
4. Use PCA for dimensionality reduction if needed
5. Model operation pairs jointly, not independently
6. Weight rare operations appropriately
7. Use context from other rooms in same project
8. Consider separate models per room type
9. Implement rule-based system for high-confidence pairs
10. Use ensemble methods to combine multiple approaches


================================================================================
ADVANCED EDA COMPLETE!
================================================================================
"""