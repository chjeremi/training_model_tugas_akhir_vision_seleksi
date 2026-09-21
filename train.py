from ultralytics import YOLO

# 1. Load pretrained YOLO-Pose model (model kecil agar cepat & ringan)
model = YOLO("yolov8n-pose.pt")  # atau 'yolo11n-pose.pt'

# 2. Latih model dengan dataset Roboflow
results = model.train(
    data="dataset/data.yaml",  # Path ke file data.yaml hasil ekstrak
    epochs=50,  # 50 epoch sudah cukup untuk tes awal
    imgsz=640,
    batch=16,
    name="cube_pose_model",
)

print(
    "\nTraining Selesai! Model terbaik disimpan di: runs/pose/cube_pose_model/weights/best.pt"
)