import cv2
import cx as cx
import numpy as np
from ultralytics import YOLO
import cvzone
import math
from sort import *

cap = cv2.VideoCapture("../videos/cars.mp4")

model = YOLO("../yolo-Weight/yolov8l.pt")
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag" "tie", "suitcase", "frisbee", "skis", "snowboard",
              "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
              "sandwich"
              "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
              "sofa", "potted plant", "bed",
              "dining-table", "toilet", "monitor", "Laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
              "hair drier", "toothbrush", "sticky notes", "Pen", "Guitar"]

# MASK TO GET THE ONLY TARGETED REGION DETECTED

mask = cv2.imread("mask.png")

# SORTING AND TRACKING
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limits = [400, 297, 673, 297]
totalCounts = []


while True:
    success, img = cap.read()

    imgRegion = cv2.bitwise_and(img, mask)


    imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (0, 0))

    results = model(imgRegion, stream=True)

    detection = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2 - x1, y2 - y1

            conf = math.ceil((box.conf[0] * 100)) / 100
            # CLASS NAME
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "car" or currentClass == "bus" or currentClass == "truck" \
                    or currentClass == "motorbike" and conf >= 0.4:
                # cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(40, y1)), scale=1, offset=10,
                #                    thickness=1)
                # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)

                currentArray = np.array([x1, y1, x2, y2, conf])
                detection = np.vstack((detection, currentArray))

    resultsTracker = tracker.update(detection)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (255, 0, 0), 5)

    for results in resultsTracker:
        x1, y1, x2, y2, id = results
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(results)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(0, 255, 0))
        cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)), scale=2, offset=10,
                           thickness=3)

        cx, cy = x1+w//2, y1 + h//2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if totalCounts.count(id) ==0:
                totalCounts.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
#    cvzone.putTextRect(img, f' Count: {len(totalCounts)}', (50, 50))
    cv2.putText(img, str(len(totalCounts)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 0), 8)

    cv2.imshow("Image", img)


    if cv2.waitKey(1) == ord('q'):  # Press 'q' to exit the loop
        break
# cv2.imshow("ImageRegion", imgRegion)