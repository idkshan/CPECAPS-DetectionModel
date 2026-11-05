"""Sections 1-4 - one module per monitored behaviour.

Each module answers two questions for ONE behaviour, from the numbers that
Stage 2 measured (it never looks at pixels):

  1. frame_state()  Is the sign present in THIS frame?   (thresholds on AI evidence)
  2. update()       Is it a real behaviour, and how much of it recently?
                    (persistence in seconds, counts and ratios over windows)
  3. severity()     How serious is it? -> score 0..1 for the risk model

    yawning.py     mouth wide open >= 1.5 s                     -> yawn events, yawns per 5 min
    gaze.py        head turned away from the driver's forward pose >= 2 s
                                                                -> gaze deviation, off-road ratio
    phone.py       phone held near the driver >= 3 s            -> phone use
    drowsiness.py  eyes closed (blinks, long closures, PERCLOS) -> eye-closure events and the
                                                                   drowsiness STATE of Table 1.11
    quality.py     is the frame good enough to judge at all?    (camera stalls, tiny faces)

The behaviours are evaluated in the order quality -> gaze -> phone -> yawning
-> drowsiness, because drowsiness must know about the others: the eyes are
NOT judged while the head is turned away, a phone is in view or the mouth is
yawning (lowered eyelids / squinting would otherwise look like closed eyes).
That is how one behaviour is kept from triggering another.

``dms.classification.classifier.BehaviorClassifier`` runs the modules for
every frame; ``dms.severity.SeverityGrader`` collects their severities.
"""

from .drowsiness import DrowsinessDetector
from .gaze import GazeDetector
from .phone import PhoneDetector
from .quality import FrameQuality
from .yawning import YawningDetector

__all__ = ["DrowsinessDetector", "FrameQuality", "GazeDetector", "PhoneDetector", "YawningDetector"]
