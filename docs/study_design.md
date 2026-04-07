# Study Design

## Design
Retrospective cohort study using longitudinal EHR and linked SDOH data.

## Index Date
First documented ART dispensing.

## Observation Window
Two years before the index date.

## Prediction Window
One year after the index date.

## Temporal Structure
Rolling yearly train-test splits were used to evaluate temporal generalizability and mimic prospective deployment.

## Outcome Definition
MPR = total ART days supplied / total days in the 1-year follow-up period

- Adherent: MPR ≥ 0.80
- Non-adherent: MPR < 0.80

Values greater than 1.0 were truncated to 1.0.
