import cv2
import mediapipe as mp


cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    result = hands.process(img_RGB)
    
    mpList = []
    
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            
            for id, lm in enumerate(handLms.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                mpList.append([id, cx, cy])
    
    if len(mpList) != 0:
        
        if mpList[4][1] > mpList[20][1]:
                cv2.putText(img, str("Right Hand"), (35, 125), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)
        else:
                cv2.putText(img, str("Left Hand"), (35, 125), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)
        
    
    cv2.imshow("img", img)
    
    if cv2.waitKey(1) &0xFF == ord("q"): break

















cv2.destroyAllWindows()
cap.release()
