import cv2
import numpy as np
import mediapipe as mp
import time
import pyautogui as gui
import math
x1=x2=y1=y2=0
webcam = cv2.VideoCapture(0)
my_hands = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils

pTime = 0
while True:
    ret, img = webcam.read()
    if not ret:
        break

    frame_height, frame_width, _ = img.shape
    img = cv2.flip(img, 1)

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    output = my_hands.process(rgb_img)
    hands = output.multi_hand_landmarks
    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(img, hand)
            landmark_list = hand.landmark
            for id, landmark in enumerate(landmark_list):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)

                # Example: draw circle on landmark 4 (thumb tip)
                if id == 8:
                    cv2.circle(img,(x,y), 8, (255, 0, 255), cv2.FILLED)
                    x1=x
                    y1=y
                if id==4:
                    cv2.circle(img, (x, y), 8, (0, 0, 255), cv2.FILLED)
                    x2=x
                    y2=y
        dist = math.hypot(x2 - x1, y2 - y1)
        cv2.line(img,(x1,y1),(x2,y2),(255,0,0),2)
        if dist > 50:
            gui.press("volumeup")
        else:  
            gui.press("volumedown")
                    
                    
                    # cv2.circle(img, (x, y), 8, (255, 0, 255), cv2.FILLED)

    cv2.imshow("Hand Volume Control", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()