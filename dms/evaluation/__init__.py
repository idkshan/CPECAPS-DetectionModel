"""Section 10 - performance evaluation (detection accuracy and alarm behaviour).

Everything that turns a session log + human labels into thesis numbers lives
here, so the scripts (``evaluate_behaviors.py``, ``evaluate_study.py``,
``benchmark_logic.py``) only parse arguments and print.

    labels.py      read and validate the ground-truth CSV files
    metrics.py     frame / segment / one-to-one event metrics, confusion matrices
    drowsiness.py  Alert / Slightly / Moderately Drowsy (thesis Table 1.11) metrics
    alarms.py      unnecessary detections, escalations, alarms and pushes per hour
    session.py     evaluate one session log; pool many sessions; Markdown tables
"""
