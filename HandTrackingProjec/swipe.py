import cv2
import numpy as np
import mediapipe as mp
import time
import pyautogui as gui
from collections import deque

gui.FAILSAFE = True
gui.PAUSE = 0  # Disabled pause to prevent camera freezing during continuous scrolling

width = 640
height = 480
capture_hands = mp.solutions.hands.Hands(max_num_hands=1)
drawing_option = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Variables for swipe and scroll
swipe_threshold = 70
swipe_ratio = 1.2
swipe_cooldown = 0.5
scroll_speed_factor = 40.0
neutral_zone_top = 180
neutral_zone_bottom = 300

history = deque(maxlen=10)
last_action_time = 0
swipe_locked = False
feedback_text = ""
feedback_color = (0, 0, 0)
feedback_timer = 0
prev_y = None
pTime = 0

print("Swipe and Scroll Controller Started.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = capture_hands.process(rgb)
    hands = output.multi_hand_landmarks
    
    mode = "Idle"
    current_time = time.time()
    
    if feedback_timer > 0:
        feedback_timer -= 1
        
    if hands:
        for hand in hands:
            drawing_option.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
            landmark_list = hand.landmark
            
            # Get finger states
            index_up = landmark_list[8].y < landmark_list[6].y
            middle_up = landmark_list[12].y < landmark_list[10].y
            ring_up = landmark_list[16].y < landmark_list[14].y
            pinky_up = landmark_list[20].y < landmark_list[18].y
            
            cx = int(landmark_list[9].x * width)
            cy = int(landmark_list[9].y * height)
            
            # Scroll Mode (Index & Middle fingers UP, others DOWN)
            if index_up and middle_up and not ring_up and not pinky_up:
                mode = "Scroll Mode"
                swipe_locked = False
                
                # Joystick-style scrolling: hold hand high to scroll up, low to scroll down
                if cy < neutral_zone_top:
                    scroll_amount = int((neutral_zone_top - cy) * scroll_speed_factor)
                    gui.scroll(scroll_amount)
                elif cy > neutral_zone_bottom:
                    scroll_amount = int((neutral_zone_bottom - cy) * scroll_speed_factor)
                    gui.scroll(scroll_amount)
                    
                prev_y = cy
                history.clear()
                
            # Swipe Mode (All 4 fingers UP)
            elif index_up and middle_up and ring_up and pinky_up:
                mode = "Swipe Mode"
                prev_y = None
                history.append((cx, cy, current_time))
                
                if not swipe_locked and current_time - last_action_time > swipe_cooldown:
                    if len(history) >= 5:
                        old_x, old_y, old_t = history[0]
                        dx = cx - old_x
                        dy = cy - old_y
                        
                        # Check if it was a quick horizontal movement
                        if abs(dx) > swipe_threshold and abs(dx) > swipe_ratio * abs(dy):
                            if dx < -swipe_threshold:
                                print("Swipe Left Detected! (Next)")
                                feedback_text = "SWIPE LEFT: NEXT"
                                feedback_color = (0, 255, 0)
                                feedback_timer = 20
                                gui.press('right')
                                gui.press('nexttrack')
                                gui.hotkey('ctrl', 'win', 'right')  # Switch to next virtual desktop
                                last_action_time = current_time
                                swipe_locked = True
                                history.clear()
                            elif dx > swipe_threshold:
                                print("Swipe Right Detected! (Prev)")
                                feedback_text = "SWIPE RIGHT: PREV"
                                feedback_color = (0, 165, 255)
                                feedback_timer = 20
                                gui.press('left')
                                gui.press('prevtrack')
                                gui.hotkey('ctrl', 'win', 'left')   # Switch to previous virtual desktop
                                last_action_time = current_time
                                swipe_locked = True
                                history.clear()
            else:
                mode = "Idle"
                prev_y = None
                history.clear()
                swipe_locked = False
                
            color = (255, 0, 255) if mode == "Scroll Mode" else ((0, 255, 255) if mode == "Swipe Mode" else (128, 128, 128))
            cv2.circle(frame, (cx, cy), 10, color, cv2.FILLED)
            
            if mode == "Scroll Mode":
                cv2.line(frame, (0, neutral_zone_top), (width, neutral_zone_top), (255, 0, 255), 2)
                cv2.line(frame, (0, neutral_zone_bottom), (width, neutral_zone_bottom), (255, 0, 255), 2)
                cv2.putText(frame, "SCROLL UP", (10, neutral_zone_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                cv2.putText(frame, "SCROLL DOWN", (10, neutral_zone_bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            
            if len(history) > 1 and mode == "Swipe Mode":
                for i in range(1, len(history)):
                    pt1 = (int(history[i-1][0]), int(history[i-1][1]))
                    pt2 = (int(history[i][0]), int(history[i][1]))
                    cv2.line(frame, pt1, pt2, (0, 255, 255), 3)
    else:
        prev_y = None
        history.clear()
        
    cv2.rectangle(frame, (0, 0), (width, 50), (20, 20, 20), cv2.FILLED)
    cv2.putText(frame, f"MODE: {mode}", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cTime = time.time()
    fps = int(1.0 / (cTime - pTime)) if (cTime - pTime) > 0 else 0
    pTime = cTime
    cv2.putText(frame, f"FPS: {fps}", (width - 120, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    if feedback_timer > 0 and feedback_text:
        cv2.rectangle(frame, (width // 2 - 200, height - 70), (width // 2 + 200, height - 20), (10, 10, 10), cv2.FILLED)
        cv2.rectangle(frame, (width // 2 - 200, height - 70), (width // 2 + 200, height - 20), feedback_color, 2)
        text_size = cv2.getTextSize(feedback_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        tx = width // 2 - text_size[0] // 2
        ty = height - 45 + text_size[1] // 2
        cv2.putText(frame, feedback_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, feedback_color, 2)
        
    cv2.imshow("Hand Swipe & Scroll Control", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
