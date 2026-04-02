# PreProcessing-2nd_Sem-HotelBooking

This contains a colab link analysing the hotel dataset, a src directory which will contain utils.py that has my pipelines and helper functions, and a report directory that has the Outputs and Graphs explained in a Report PDF

# Feature Engineering Capstone

## Overview
This project focuses on feature engineering, transformation, and selection techniques to improve predictive performance for hotel booking cancellation prediction.

## Repository Structure

```
FeatureEngineering_Capstone.ipynb   # Main notebook
/src
    utils.py                        # Helper functions
/report
    Report.pdf                      # Final report with analysis
```

## How to Run

### Option 1: Google Colab
- Open the notebook using the provided Colab link (https://colab.research.google.com/drive/1Tm8O0WdDHlk4n_IdF9XMPfQpAOUggve0?usp=sharing)
- Run all cells from top to bottom

### Option 2: Local Execution

1. Clone the repository:
```
git clone https://github.com/shivjain123/PreProcessing-2nd_Sem-HotelBooking
cd PreProcessing-2nd_Sem-HotelBooking
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Run the notebook:
```
jupyter notebook FeatureEngineering_Capstone.ipynb
```

## Key Steps

- Data preprocessing and cleaning
- Feature transformation and scaling
- Feature extraction (temporal & behavioural)
- Feature construction (ratios, interactions, aggregations)
- Feature importance analysis (RF, MI, Chi-Square)
- Feature selection
- Model evaluation and comparison

## Results Summary

| Stage | ROC-AUC | F1 Score |
|------|--------|---------|
| Baseline | 0.8917 | 0.7588 |
| After Preprocessing | 0.9009 | 0.7701 |
| After Feature Engineering | 0.9365 | 0.8141 |
| After Feature Selection | 0.9356 | 0.8104 |

## Notes

- Feature engineering significantly improved model performance
- Feature selection reduced dimensionality with minimal performance loss
- RobustScaler was effective due to the presence of outliers

## Author

Shiv Jain
