import cv2
import time
import mediapipe as mp


cap = cv2.VideoCapture(0)

mphands = mp.solutions.hands
hands = mphands.Hands()

mpDraw = mp.solutions.drawing_utils

cTime = 0
pTime = 0

while True:
    success, img = cap.read()
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    result = hands.process(img_RGB)
    print(result.multi_hand_landmarks)
    
    if result.multi_hand_landmarks:
        for handlms in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handlms, mphands.HAND_CONNECTIONS)
            
            for id, lm in enumerate(handlms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                
                if id == 20:
                    cv2.circle(img, (cx, cy), 9, (255, 0, 0), cv2.FILLED)
    # fps
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, "FPS: " + str(int(fps)), (0, 75), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 5)
    
    cv2.imshow("img", img)
    if cv2.waitKey(1) &0xFF == ord("q"): break
    

cap.release()
cv2.destroyAllWindows()