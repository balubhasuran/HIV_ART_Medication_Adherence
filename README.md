# Machine Learning–Based Prediction of Antiretroviral Therapy Adherence in Young Adults With HIV Using Longitudinal Clinical and Social Determinants of Health Data

## Overview
This repository contains the analytical code and project structure for a study developing machine learning models to predict antiretroviral therapy (ART) adherence among young adults living with HIV using longitudinal clinical data and social determinants of health (SDOH).

The study uses real-world electronic health record (EHR) and linked contextual data from the OneFlorida+ Clinical Research Network to predict medication adherence during the first year after ART initiation.

## Study Objective
To develop and evaluate a machine learning framework for predicting ART adherence among young adults with HIV using longitudinal clinical, behavioral, laboratory, and SDOH data.

## Key Study Details
- **Population:** Young adults aged 18–29 years with HIV initiating ART
- **Cohort size:** 3,246 patients
- **Data source:** OneFlorida+ Clinical Research Network
- **Outcome:** Medication Possession Ratio (MPR) over 1 year
- **Adherence definition:** Adherent if MPR ≥ 0.80
- **Observation window:** 2 years before index date
- **Prediction window:** 1 year after index date
- **Temporal design:** Rolling yearly train-test splits from 2012–2022
- **Models evaluated:**
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - XGBoost
  - LightGBM
  - LSTM
- **Evaluation methods:**
  - ROC/AUC
  - Calibration analysis
  - Decision Curve Analysis (DCA)
  - SHAP interpretation

## Main Findings
- LightGBM achieved the strongest predictive performance.
- Average AUC was approximately 0.62, with a peak AUC of 0.79.
- Behavioral health and SDOH variables were more informative than traditional comorbidities.
- Important predictors included hemoglobin, age, psychiatric disorders, and contextual SDOH features.
- Decision curve analysis showed improved net clinical benefit across risk thresholds of 0.10–0.40.

## Repository Structure
- `data/` : raw, processed, and external linked datasets
- `docs/` : study documentation and figure/table descriptions
- `notebooks/` : exploratory and modeling notebooks
- `src/` : reusable code for preprocessing, modeling, evaluation, and interpretation
- `outputs/` : generated figures, tables, and trained model artifacts
- `tests/` : basic tests for pipeline validation

## Data Availability
The underlying patient-level data are not publicly shareable because of privacy, institutional, and IRB restrictions.

Data may be made available from the corresponding author upon reasonable request, pending institutional approval, IRB requirements, and an executed data use or data sharing agreement.

## Code Availability
This repository contains the code and project structure used for preprocessing, feature engineering, model training, evaluation, and interpretation.

## Reproducibility Notes
Because the source EHR data are not public, users must adapt the pipeline to their local secure environment and available data schema.

## Suggested Citation
Bhasuran B, Hall A, Addo P, Liu Y, MacDonell K, Naar S, He Z. Machine Learning–Based Prediction of Antiretroviral Therapy Adherence in Young Adults With HIV Using Longitudinal Clinical and Social Determinants of Health Data.

## Contact
**Balu Bhasuran**  
School of Information, Florida State University  
Tallahassee, FL, USA

## License
Add your preferred license before public release.
