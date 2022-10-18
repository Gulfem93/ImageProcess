import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mpHand = mp.solutions.hands
hands = mpHand.Hands()
mpDraw = mp.solutions.drawing_utils

mlIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    result = hands.process(img_RGB)
    
    lmList = []
    
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHand.HAND_CONNECTIONS)
            
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                
                lmList.append([id, cx, cy])
                
    
    if len(lmList) != 0:
        fingers = []
        
        if lmList[mlIds[0]][1] < lmList[mlIds[0] - 2][1]:
            fingers.append(1)
        else:
            fingers.append(0)
            
        for id in range(1, 5):
            
            if lmList[mlIds[id]][2] < lmList[mlIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        fingersT = fingers.count(1)
    
        cv2.putText(img, str(fingersT), (30, 125), cv2.FONT_HERSHEY_PLAIN, 5, (255, 5, 5), 5)
                
    
    cv2.imshow("img", img)
    
    if cv2.waitKey(1) &0xFF == ord("q"): break

cv2.destroyAllWindows()
cap.release()


















