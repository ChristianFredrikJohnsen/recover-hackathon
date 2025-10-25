# Baseline Model Summary

## Overview
Created minimal baseline models for the Recover Hackathon to predict missing work operations in rooms.

## Models Created

### 1. Simple Rule-Based Baseline (`baseline_simple.py`)
**Validation Score: -0.5884** (negative = worse than predicting empty)

**Approach:**
- Learns co-occurrence patterns from training data
- For each visible operation, predicts operations that frequently co-occur
- Uses room-specific patterns when available

**Problems:**
- Too many false positives (FP cost = -0.25 each)
- Doesn't account for rooms that should have no predictions
- Not selective enough

---

### 2. Improved Rule-Based Baseline (`baseline_improved.py`) ✅
**Validation Score: 0.0661** (6.61% improvement over dummy baseline)

**Approach:**
- Conservative prediction strategy
- Only predicts when confidence is high (P(hidden | visible) > 0.2)
- Predicts empty lists for room types that often have no missing operations
- Requires minimum training examples before making predictions
- Weights room-specific patterns higher than global patterns

**Key Statistics:**
- Average predictions per room: 1.18
- Empty predictions: 53% of rooms
- Max predictions per room: 5

**Performance:**
This baseline achieves ~6.6% score on the normalized scale where:
- 0% = predicting all empty lists (dummy baseline)
- 100% = perfect predictions

---

### 3. XGBoost Baseline (`baseline_xgboost.py`)
**Status:** Code ready but not run yet (takes longer to train)

**Approach:**
- Train separate XGBoost classifier for each operation (multi-label classification)
- Features: visible operations + room type + metadata (insurance, distance, date)
- Currently set to train only top 50 operations for speed

**To run:**
```bash
uv run python baseline_xgboost.py
```

---

## Dataset Insights (from EDA)

### Key Patterns:
1. **Strong co-occurrence patterns**: "Riv gulvlist" (remove floor trim) → "Ny gulvlist" (new floor trim) in 25.64% of rooms
2. **Room-specific operations**: Each room type has characteristic operations
3. **Sparse data**: 97% sparsity, average 10.28 operations per room
4. **High-confidence rules**: 80 rules with P(B|A) > 0.8 found

### Most Common Operations:
1. Light floor protection (68,603 occurrences)
2. Construction cleaning (59,512)
3. Internal furniture moving (59,020)
4. New floor trim (52,446)
5. Remove floor trim (51,740)

---

## Submission Files Created

All submissions are saved in `submissions/` folder:

1. `submission_20251025_134307.csv` - Simple baseline (negative score)
2. `submission_20251025_134526.csv` - Improved baseline (0.0661 score) ✅

---

## Next Steps to Improve

### Short-term (Quick Wins):
1. **Tune hyperparameters** of improved baseline:
   - Try different `min_confidence` thresholds (currently 0.2)
   - Adjust `max_predictions` (currently 5)
   - Test different weighting schemes

2. **Use more training data**:
   - Currently using 20% of training data for speed
   - Try with 50-100% of data

3. **Add more sophisticated features**:
   - Count of visible operations
   - Specific operation combinations (e.g., if both "Riv" and "Ny" operations present)
   - Context from other rooms in same project

### Medium-term (ML Models):
4. **Complete XGBoost implementation**:
   - Train for all 388 operations (not just top 50)
   - Tune hyperparameters
   - Try different thresholds for predictions

5. **Try other ML approaches**:
   - Random Forest
   - Logistic Regression with feature engineering
   - Neural networks

6. **Ensemble methods**:
   - Combine rule-based + ML predictions
   - Voting between multiple models

### Advanced (Best Performance):
7. **Use project context**:
   - Leverage operations from other rooms in same project
   - Model dependencies between rooms

8. **Specialized models per room type**:
   - Separate model for kitchen, bathroom, bedroom, etc.

9. **Calibration**:
   - Better calibrate prediction probabilities
   - Optimize for the specific scoring metric (TP/FP/FN weights)

10. **Feature engineering based on domain knowledge**:
    - Detect "Riv" → "Ny" patterns automatically
    - Use operation naming patterns (prefix analysis)
    - Incorporate insurance company preferences

---

## How to Use

### Run improved baseline (recommended):
```bash
python3 baseline_improved.py
```

### Run XGBoost baseline (if uv environment is set up):
```bash
uv run python baseline_xgboost.py
```

### Submit to Kaggle:
1. Go to https://www.kaggle.com/competitions/hackathon-recover-x-cogito/submit
2. Upload the submission CSV from `submissions/` folder
3. Compare scores with validation score

---

## Technical Details

### Scoring Function:
- True Positive (TP): +1 point
- False Positive (FP): -0.25 points
- False Negative (FN): -0.5 points
- Empty room predicted correctly: +1 point

Normalized to 0-100% scale where:
- 0% = dummy baseline (predict all empty)
- 100% = perfect predictions

### Data Format:
- Training: 51,009 projects, 185,635 rooms
- Validation: 3,000 projects, 10,827 rooms
- Test: 5,994 projects, 18,299 rooms (for submission)

Each room has:
- Visible operations (X)
- Hidden operations to predict (Y)
- Room type (kitchen, bathroom, etc.)
- Project metadata (insurance company, location, date)

---

## Files in This Repository

```
baseline_simple.py          # First attempt (negative score)
baseline_improved.py        # Conservative approach (6.61% score) ✅
baseline_xgboost.py         # ML approach (not run yet)
BASELINE_SUMMARY.md         # This file
submissions/                # Generated submission files
  ├── submission_20251025_134307.csv
  └── submission_20251025_134526.csv
```

---

## Conclusion

Successfully created a working baseline that achieves positive validation score (6.61%).

The key insight: **Being conservative is better than being aggressive** due to the scoring function heavily penalizing false positives and false negatives.

Ready for iteration and improvement! 🚀
