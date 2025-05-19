from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("250519_n_detection.pt")

# Export the model to TensorRT format
model.export(format="engine")  # creates 'yolo11n.engine'

# Load the exported TensorRT model
tensorrt_model = YOLO("yolo11n.engine")

# Run inference
results = tensorrt_model("https://ultralytics.com/images/bus.jpg")