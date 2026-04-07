# Manuscript Summary

## Title
Machine Learning–Based Prediction of Antiretroviral Therapy Adherence in Young Adults With HIV Using Longitudinal Clinical and Social Determinants of Health Data

## Objective
To predict ART adherence among young adults with HIV using longitudinal clinical and SDOH data.

## Data Source
OneFlorida+ Clinical Research Network.

## Cohort
- Age: 18–29 years
- Final sample: 3,246 patients

## Outcome
Medication Possession Ratio (MPR) over 1 year after ART initiation.

## Predictors
- Demographics
- Clinical comorbidities
- Laboratory values
- Psychiatric disorders
- Substance use
- Area-level SDOH

## Models
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- LSTM

## Evaluation
- AUC/ROC
- Calibration
- DCA
- SHAP

## Key Findings
LightGBM performed best. Important predictors included hemoglobin, age, psychiatric disorders, and SDOH variables. Clinical and social context mattered more than traditional comorbidity burden.
