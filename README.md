# DeFi Credit Scoring Model for Aave V2 Protocol

## Overview
This repository contains a machine learning model that assigns credit scores (0-1000) to DeFi wallets based on their historical transaction behavior on the Aave V2 protocol. The model analyzes transaction patterns to identify reliable users versus risky, bot-like, or exploitative behavior.

## Credit Scoring Logic

### Feature Engineering
The model extracts 25+ features from raw transaction data, categorized into:

#### 1. Transaction Volume & Activity
- **Total transactions**: Number of interactions with the protocol
- **Total USD volume**: Sum of all transaction values
- **Average/Median transaction size**: Statistical measures of transaction amounts
- **Transaction frequency**: Transactions per day over activity period

#### 2. Behavioral Patterns
- **Action ratios**: Proportion of deposits, borrows, repays, redeems, liquidations
- **Repay-to-borrow ratio**: Key indicator of responsible borrowing (>0.8 = excellent)
- **Borrow-to-deposit ratio**: Leverage indicator (>2 = high risk)
- **Asset diversity**: Number of different assets used

#### 3. Risk Indicators
- **Liquidation history**: Any liquidation events significantly lower scores
- **Transaction consistency**: Coefficient of variation in transaction sizes
- **Time patterns**: Average time between transactions (bot detection)

#### 4. Temporal Features
- **Activity duration**: Days between first and last transaction
- **Transaction timing**: Patterns that may indicate automated behavior

### Scoring Algorithm

The model uses a two-stage approach:

#### Stage 1: Rule-Based Baseline (Training Data Generation)
Starting from a base score of 500, the algorithm applies:

**Positive Adjustments (+)**
- High repayment ratio (≥0.8): +100 points
- Long-term activity (>30 days): +50 points
- Asset diversification (>2 assets): +30 points
- High activity (>10 transactions): +40 points
- Good deposit behavior (>30% deposits): +60 points

**Negative Adjustments (-)**
- Any liquidation events: -200 points
- High leverage (borrow/deposit >2): -100 points
- Bot-like frequency (>50 tx/day): -80 points
- Inconsistent behavior (high CV): -40 points
- Poor repayment (<0.2 ratio): -150 points

#### Stage 2: Machine Learning Refinement
A Random Forest model learns complex patterns from the rule-based scores and features to:
- Capture non-linear relationships
- Handle feature interactions
- Improve prediction accuracy
- Generalize beyond rule-based logic

### Score Interpretation

| Score Range | Risk Category | Description |
|-------------|---------------|-------------|
| 800-1000 | Excellent | Highly reliable, consistent repayment, diversified usage |
| 600-799 | Low Risk | Good behavior, regular activity, minimal risk indicators |
| 300-599 | Medium Risk | Mixed signals, some risk factors present |
| 0-299 | High Risk | Risky behavior, liquidations, bot-like patterns |

## Usage

### Installation
```bash
pip install -r requirements.txt
```

### Running the Model
```bash
python defi_credit_scorer.py
```

### Example Output
```
Total Wallets Analyzed: 3,497
Average Credit Score: 542.3
Median Credit Score: 558.0

Risk Distribution:
  Low Risk: 1,398 wallets (40.0%)
  Medium Risk: 1,189 wallets (34.0%)
  High Risk: 629 wallets (18.0%)
  Excellent: 281 wallets (8.0%)
```

## Model Performance
- **Feature Importance**: Repayment behavior, liquidation history, and activity patterns are the strongest predictors
- **Validation**: Rule-based baseline ensures logical consistency
- **Extensibility**: Easy to add new features or adjust scoring logic

## Key Features
- **Comprehensive**: Analyzes 25+ behavioral and risk indicators
- **Transparent**: Clear scoring logic with interpretable results
- **Robust**: Handles edge cases and missing data
- **Scalable**: Processes large datasets efficiently
- **Extensible**: Modular design for easy feature addition

## Architecture
The `DeFiCreditScorer` class provides:
- Data loading and preprocessing
- Feature extraction and engineering
- Model training and validation
- Score generation and interpretation
- Comprehensive reporting

This model provides a solid foundation for DeFi credit assessment while maintaining transparency and extensibility for future enhancements.