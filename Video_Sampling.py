import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture('DogAndMan.mp4')
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

model = YOLO("yolov8n.pt")
count = 0
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("out.mp4", fourcc, 30, (width, height))

for frame in list_of_frames:
    results = model.predict(source=frame)
    res = results[0]
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        cls = int(box.cls)
        label = model.names[cls]
        if (cls == 0):
             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
             cv2.imwrite(f"dump/men/{count}.jpg", frame)
        if (cls == 16):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.imwrite(f"dump/dog/{count}.jpg", frame)
        count+=1
    out.write(frame)
        
