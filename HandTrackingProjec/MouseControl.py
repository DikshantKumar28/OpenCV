import cv2
import mediapipe as mp
import pyautogui as gui
import time
# import autopy as ap
import math
import numpy as np

capture_hands=mp.solutions.hands.Hands()
drawing_option=mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
width = 720
height = 540
cap.set(cv2.CAP_PROP_FRAME_WIDTH,width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
screen_widht,screen_height=gui.size()
x1=y1=x5=y5=x0=y0=0
mouse_x=mouse_y=0
is_tapped = False
plocX, plocY = 0, 0
clocX, clocY = 0, 0
smoothening = 8
frameR = 100 # Frame reduction margin
while True:
    ret, frame = cap.read()
    
    frame = cv2.flip(frame,1)
    cv2.rectangle(frame, (frameR, frameR), (width - frameR, height - frameR), (255, 0, 255), 2)
    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    output = capture_hands.process(rgb)
    hands = output.multi_hand_landmarks
    if hands:
        for hand in hands:
            drawing_option.draw_landmarks(frame,hand)
            landmark_list = hand.landmark
            for id, landmark in enumerate(landmark_list):


                x = int(landmark.x * width)
                y = int(landmark.y * height)
                if id == 0:
                    x0=x
                    y0=y
                if id == 5:
                    x5=x
                    y5=y
                if id == 8:
                    mouse_x = np.interp(x, (frameR, width - frameR), (0, screen_widht))
                    mouse_y = np.interp(y, (frameR, height - frameR), (0, screen_height))
                    x1=x
                    y1=y
                    cv2.circle(frame, (x, y), 8, (255, 0, 255), cv2.FILLED)
            
            palm_size = math.hypot(x5 - x0, y5 - y0)
            index_length = math.hypot(x5 - x1, y5 - y1)
            
            if palm_size > 0:
                ratio = index_length / palm_size
                if ratio < 0.75:
                    if not is_tapped:
                        is_tapped = True
                        gui.click()
                    cv2.circle(frame, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
                else:
                    is_tapped = False
                    clocX = plocX + (mouse_x - plocX) / smoothening
                    clocY = plocY + (mouse_y - plocY) / smoothening
                    gui.moveTo(clocX, clocY)
                    plocX, plocY = clocX, clocY
                
    cv2.imshow("Hand Mouse Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()