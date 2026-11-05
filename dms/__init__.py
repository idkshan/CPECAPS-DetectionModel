"""AI-based adaptive driver-monitoring system - detection model (version 3).

Sections (see docs/ARCHITECTURE.md for inputs, outputs and parameters of each):

    1-4  behaviours        dms.behaviors        yawning, gaze, phone, drowsiness (+ frame quality)
    5    risk              dms.severity, dms.risk   severity per behaviour -> fuzzy risk level
    6    alerts            dms.alerts            alert state machine, buzzer / push commands
    7    datasets          dms.data              group splits, dataset audit (training side)
    8    camera / video    dms.camera            frame sources, stream health, frame timing
    9    context           dms.context           GPS speed, motion, driving duration
    10   integration       dms.monitor, dms.engine, dms.session_log, dms.evaluation

The AI models run in dms.perception (MediaPipe Face Landmarker, YOLO, optional
eye-state CNN); dms.features turns their outputs into numbers (EAR, MAR, head
pose); dms.classification holds the per-driver calibration and the shared
evidence / timing helpers used by the behaviour modules.
"""

__version__ = "3.0.0"
