import numpy as np
import cv2
import mediapipe as mp
import pickle
import matplotlib.pyplot as plt
import time

load_model = pickle.load(open("signModel.hd5","rb"))

mphands = mp.solutions.hands
hands = mphands.Hands(static_image_mode=False,
                      max_num_hands=1,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

_, frame = cap.read()

stop_time = 0.2
h, w, c = frame.shape
prev_time = time.time()
last_pred = ""
np.set_printoptions(linewidth=np.inf)

def show_frame(frame):
    cv2.imshow("Frame", frame)

while True:
    x_max = 0
    y_max = 0
    x_min = w
    y_min = h
    curr_time = time.time()
    _, frame = cap.read()
    framergb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks
    
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    if hand_landmarks:
        for handLMs in hand_landmarks:
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            hand_w = x_max - x_min
            hand_h = y_max - y_min
            if hand_w > hand_h:
                diff = hand_w-hand_h
                y_min -= int(diff/2)
                y_max += int(diff/2)
            else:
                diff = hand_h-hand_w
                x_min -= int(diff/2)
                x_max += int(diff/2)
            cv2.rectangle(frame, (x_min-20, y_min-20), (x_max+20, y_max+20), (255, 0, 0), 2)#             mp_drawing.draw_landmarks(frame, handLMs, mphands.HAND_CONNECTIONS)
        curr_time = time.time()
        if curr_time - stop_time > prev_time:
            prev_time = curr_time
            crop_img = framergb[y_min-20: y_max+20, x_min-20: x_max+20]
            #crop_img = framergb[y_min: y_max, x_min: x_max]
    #         if(len(crop_img)):
    #             (row, col) = crop_img.shape[0:2]
    #             # Take the average of pixel values of the BGR Channels
    #             # to convert the colored image to grayscale image
    #             for i in range(row):
    #                 for j in range(col):
    #                     # Find the average of the BGR pixel values
    #                     crop_img[i, j] = sum(crop_img[i, j]) * 0.33
    #             crop_img = crop_img.flatten()
            try:
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY) 
                crop_img = cv2.GaussianBlur(crop_img, (7, 7), 0)
                crop_img = cv2.resize(crop_img, (28, 28))
                #cv2.imwrite('crop_image0.jpg', crop_img)
        #         print(crop_img.shape)
                crop_img = crop_img.reshape(1,28,28,1)
                #print(crop_img.shape)
                crop_img = np.array(crop_img, dtype='float')
                crop_img /= 255
                y = load_model.predict(crop_img, verbose=0)
                pred = np.argmax(y)
                if(pred >= 9):
                    pred += 1
                last_pred = chr(pred+ord("A"))
                #print(crop_img.round(1).squeeze())
                #time.sleep(1)
                #plt.imshow(crop_img[0])
                #plt.show()
            except:
                pass
    print("Detected: ", last_pred)
    cv2.putText(frame, last_pred, (x_min-5,y_min-25), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 3)
    cv2.imshow("Frame", frame)
    
cap.release()
cv2.destroyAllWindows()