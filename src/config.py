"""
Project configuration settings.
"""

RANDOM_STATE = 42
ADHERENCE_THRESHOLD = 0.80

OBSERVATION_WINDOW_YEARS = 2
PREDICTION_WINDOW_YEARS = 1

TARGET_COLUMN = "adherence_binary"
ID_COLUMN = "patient_id"
INDEX_DATE_COLUMN = "index_date"
