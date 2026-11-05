# models/

Model files are not stored in git. Download them with:

```bash
python scripts/download_models.py          # face_landmarker.task + yolov8n / yolo11n / yolo26n
python scripts/download_models.py --cls    # + YOLO classification weights (eye-state training)
```

| File | What it is | Used by |
|---|---|---|
| `face_landmarker.task` | MediaPipe Face Landmarker bundle (face detector + 478-point mesh + 52 blendshapes) | Stage 1 |
| `yolov8n.pt` | YOLOv8-nano, COCO-pretrained (paper baseline) | Stage 1 phone detector |
| `yolo11n.pt` | YOLO11-nano, COCO-pretrained | Stage 1 phone detector (comparison) |
| `yolo26n.pt` | YOLO26-nano, COCO-pretrained | Stage 1 phone detector (comparison) |
| `*-cls.pt` | YOLO classification backbones | starting point for the optional eye-state model |
