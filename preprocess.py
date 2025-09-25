import cv2
from ultralytics import YOLO
import numpy as np
import onnxruntime

def preprocess(image, input_height, input_width):
    original_height, original_width = image.shape[:2]
    # 1. Resize the image to the model's input size, maintaining aspect ratio.
    scale = min(input_width / original_width, input_height / original_height)
    resized_width, resized_height = int(original_width * scale), int(original_height * scale)
    resized_img = cv2.resize(image, (resized_width, resized_height))
    # 2. Create a blank canvas and paste the resized image onto it (letterboxing).
    pad_x = (input_width - resized_width) / 2
    pad_y = (input_height - resized_height) / 2
    padded_img = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
    padded_img[int(pad_y):int(pad_y) + resized_height, int(pad_x):int(pad_x) + resized_width] = resized_img
    # 3. Convert from HWC to CHW format, BGR to RGB, and normalize.
    input_tensor = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
    input_tensor = input_tensor.astype(np.float32) / 255.0
    input_tensor = np.transpose(input_tensor, (2, 0, 1))
    # 4. Add a batch dimension.
    input_tensor = np.expand_dims(input_tensor, axis=0)
    return input_tensor, scale, pad_x, pad_y

def post_processing(detections, scale, pad_x, pad_y):
    CONF_THRESHOLD = 0.5
    PERSON_CLASS_ID = 0
    CAR_CLASS_ID = 2

    final_detections = []

    for det in detections:
        x1, y1, x2, y2, conf, cls = det

        # Filter detections by confidence and class
        if conf < CONF_THRESHOLD or not(int(cls) == PERSON_CLASS_ID or int(cls) == CAR_CLASS_ID):
            continue

        # ==== Accurate Coordinate Scaling ====
        # 1. Remove padding
        x1_unpad = x1 - pad_x
        y1_unpad = y1 - pad_y
        x2_unpad = x2 - pad_x
        y2_unpad = y2 - pad_y

        # 2. Rescale to original image size
        x1_final = int(x1_unpad / scale)
        y1_final = int(y1_unpad / scale)
        x2_final = int(x2_unpad / scale)
        y2_final = int(y2_unpad / scale)

        final_detections.append([x1_final, y1_final, x2_final, y2_final, conf, cls])
    return final_detections

def post_processing_v2(detections, scale, pad_x, pad_y):
    CONF_THRESHOLD = 0.5
    final_detections = detections[detections[:,4] > CONF_THRESHOLD]
    final_detections[:,0] -= pad_x
    final_detections[:,2] -= pad_x
    final_detections[:,1] -= pad_y
    final_detections[:,3] -= pad_y
    final_detections[:,0] /= scale
    final_detections[:,2] /= scale
    final_detections[:,1] /= scale
    final_detections[:,3] /= scale      
    return final_detections

def predict(session, frame):
     model_inputs = session.get_inputs()
     input_height, input_width = model_inputs[0].shape[2], model_inputs[0].shape[3]
     input_tensor, scale, pad_x, pad_y = preprocess(frame, input_height, input_width)
     input_name = model_inputs[0].name
     outputs = session.run(None, {input_name: input_tensor})[0]
     detections = post_processing_v2(outputs[0], scale, pad_x, pad_y)
     return detections

cap = cv2.VideoCapture('manAndCar.mp4')
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()
count = 0
list_of_frames = list()
height = 0; 
width = 0;
while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break
        height, width = frame.shape[:2]
        if (count % 5 == 0):
             list_of_frames.append(frame)
        count += 1
       
print(f"Actual count of frames {count}")
print(len(list_of_frames))

model = YOLO("yolo11n.pt")
count = 0
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("out.mp4", fourcc, 30, (width, height))
model = YOLO("yolo11n.pt")
model.export(format="onnx", nms=True)
session = onnxruntime.InferenceSession("yolo11n.onnx", providers=['CPUExecutionProvider'])

for frame in list_of_frames:
    results = predict(session, frame) 
    for box in results:
        x1, y1, x2, y2, score, cls = map(int, box[:6])
        label = model.names[cls]  
        if cls == 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        elif cls == 2:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        count += 1
    out.write(frame)



