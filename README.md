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
Starting from a base score of 400, the algorithm applies tiered adjustments:

**Tier 1: Excellent Behavior Indicators (Major Positive Impact)**
- Perfect repayment ratio (≥1.0): +200 points
- Very good repayment (≥0.8): +150 points
- Good repayment (≥0.5): +100 points

**Tier 2: Long-term Commitment and Diversity**
- Long-term activity (>90 days): +100 points
- Medium-term activity (>30 days): +60 points
- High asset diversity (>5 assets): +80 points
- Well diversified (>2 assets): +50 points

**Tier 3: Activity Level and Behavior**
- Very active user (>100 transactions): +80 points
- Active user (>50 transactions): +60 points
- Moderate activity (>10 transactions): +40 points
- Deposit-heavy behavior (>50% deposits): +100 points
- Good deposit behavior (>30% deposits): +60 points

**Tier 4: Volume and Consistency Bonuses**
- High volume user (>1e23 USD): +50 points
- Medium volume user (>1e22 USD): +30 points
- Very consistent transactions (CV<1): +40 points
- Consistent transactions (CV<2): +20 points

**Tier 5: Risk Penalties (Major Negative Impact)**
- Any liquidation events: -300 points
- Very high leverage (>3x): -200 points
- High leverage (>2x): -100 points
- Extreme bot-like behavior (>100 tx/day): -150 points
- Bot-like behavior (>50 tx/day): -100 points
- Very poor repayment (<0.1): -250 points
- Poor repayment (<0.2): -150 points

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
python defi_credit_scorer.py data/user-wallet-transactions.json [output.csv]
```

The script will:
1. Load and analyze transaction data
2. Extract 25+ behavioral features
3. Train the credit scoring model
4. Generate credit scores for all wallets
5. Create comprehensive analysis report (`analysis.md`)
6. Generate visualization charts (`results/wallet_analysis_charts.png`)
7. Save detailed results to CSV file

### Example Output
```
============================================================
DeFi Credit Scoring Model
============================================================
Loading data from data/user-wallet-transactions.json...
Loaded 100000 transactions from 3497 unique wallets
Extracting features...
Creating synthetic credit scores for training...
Training credit scoring model...

Model Performance:
  Mean Absolute Error: 45.23
  Root Mean Square Error: 67.81

Top 10 Most Important Features:
                    feature  importance
repay_to_borrow_ratio        0.2543
liquidation_count           0.1876
activity_duration_days      0.1234
asset_diversity             0.0987
total_transactions          0.0856

============================================================
CREDIT SCORE SUMMARY
============================================================
Total Wallets Analyzed: 3,497
Average Credit Score: 584.2
Median Credit Score: 592.0
Score Range: 89 - 943

Risk Distribution:
  Low Risk: 1,598 wallets (45.7%)
  Medium Risk: 1,189 wallets (34.0%)
  High Risk: 529 wallets (15.1%)
  Excellent: 181 wallets (5.2%)

============================================================
PERFORMING DETAILED ANALYSIS
============================================================
Analyzing wallet behavior patterns...
Creating visualizations...
Visualizations saved to results/wallet_analysis_charts.png and results/detailed_wallet_analysis.png
Generating analysis report...
Comprehensive analysis report saved to: analysis.md
```

## Generated Files

After running the model, you'll get:

1. **`analysis.md`** - Comprehensive analysis report with:
   - Executive summary and score distribution
   - Behavioral patterns across score ranges
   - High-risk and excellent wallet analysis
   - Feature importance rankings
   - Actionable recommendations

2. **`results/wallet_analysis_charts.png`** - Main visualization dashboard with:
   - Credit score distribution histogram
   - Wallet count by score ranges
   - Average USD volume by score range
   - Repay-to-borrow ratio analysis

3. **`results/detailed_wallet_analysis.png`** - Detailed behavioral analysis with:
   - Activity duration patterns
   - Asset diversity trends
   - Transaction frequency analysis
   - Action ratio comparisons
   - Risk indicator distributions

4. **`results/credit_scores.csv`** - Complete results with wallet scores and metrics

## Model Performance

- **Enhanced Scoring Range**: Now capable of generating scores from 0-1000 with wallets achieving excellent ratings (800+)
- **Improved Granularity**: Multi-tier scoring system provides more nuanced risk assessment
- **Feature Importance**: Repayment behavior, liquidation history, and activity patterns are the strongest predictors
- **Comprehensive Analysis**: Automated generation of behavioral insights and visualizations
- **Validation**: Rule-based baseline ensures logical consistency with machine learning refinement
- **Extensibility**: Easy to add new features or adjust scoring logic

## Key Features

- **Comprehensive Analysis**: Analyzes 25+ behavioral and risk indicators with automated insights
- **Transparent Scoring**: Clear multi-tier scoring logic with interpretable results
- **Visual Analytics**: Automated generation of charts and behavioral pattern analysis
- **Robust Processing**: Handles edge cases and missing data efficiently
- **Scalable Architecture**: Processes large datasets with optimized performance
- **Extensible Design**: Modular structure for easy feature addition and model enhancement
- **Detailed Reporting**: Generates comprehensive analysis reports with actionable insights

## Architecture

The enhanced `DeFiCreditScorer` class provides:

- Data loading and preprocessing with comprehensive error handling
- Feature extraction and engineering with 25+ behavioral indicators
- Multi-tier rule-based scoring system for transparent baseline generation
- Machine learning model training and validation with Random Forest
- Automated score generation and risk categorization
- Comprehensive behavioral analysis across score ranges
- Visualization generation for pattern recognition
- Detailed reporting with actionable insights and recommendations

This enhanced model provides a robust foundation for DeFi credit assessment with comprehensive analysis capabilities, maintaining transparency and extensibility for future enhancements.