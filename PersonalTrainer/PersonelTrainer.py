import cv2
import numpy as np
import mediapipe as mp
import math

cap = cv2.VideoCapture("video2.mp4")


mpPose = mp.solutions.pose
pose = mpPose.Pose()

mpDraw = mp.solutions.drawing_utils

dir = 0
count = 0

def findAndle(img, p1, p2, p3, lmList, draw = True):
    
    x1, y1 = lmList[p1][1:]
    x2, y2 = lmList[p2][1:]
    x3, y3 = lmList[p3][1:]
    
    andle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(x1 - x2, x3 - x2))
    
    if andle < 0:
        andle += 360
    
    if draw:
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.line(img, (x3, y3), (x2, y2), (0, 0, 255), 3)
        
        
        cv2.circle(img, (x1, y1), 10, (0, 255, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (0, 255, 255), cv2.FILLED)
        cv2.circle(img, (x3, y3), 10, (0, 255, 255), cv2.FILLED)
        
        cv2.circle(img, (x1, y1), 15, (0, 255, 255))
        cv2.circle(img, (x2, y2), 15, (0, 255, 255))
        cv2.circle(img, (x3, y3), 15, (0, 255, 255))
        
        cv2.putText(img, str(int(andle)), (x2 - 50, y2 + 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        
    return andle
    

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    result = pose.process(imgRGB)
    lmList = []
    
    if result.pose_landmarks:
        mpDraw.draw_landmarks(img, result.pose_landmarks, mpPose.POSE_CONNECTIONS)
        
        for id, lm in enumerate(result.pose_landmarks.landmark):
            h, w, _ = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmList.append([id, cx, cy])
    
    if len(lmList) != 0:
        # andle = findAndle(img, 11, 13, 15, lmList)
        # per = np.interp(andle, (180, 225), (0, 100))
        
        andle = findAndle(img, 23, 25, 27, lmList)
        per = np.interp(andle, (135, 225), (0, 100))
        if per == 100:
            if dir == 0:
                count += 0.5
                dir = 1
        
        if per == 0:
            if dir == 1:
                count += 0.5
                dir = 0
        print(andle)

    cv2.putText(img, str(int(count)), (30, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 3)
        
    cv2.imshow("image", img)
    cv2.waitKey(10)






























