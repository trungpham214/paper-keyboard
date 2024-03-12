import math

import cv2
import mediapipe as mp
import pyautogui
import time
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import asyncio
import string
import pytesseract
import helper_function
import keyboard

pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'

helper = helper_function.Helper()
keyboard = keyboard.Keyboard()

webcam = cv2.VideoCapture(0)
my_hands = mp.solutions.hands.Hands()
draw = mp.solutions.drawing_utils
detector = HandDetector(detectionCon=0.8, maxHands=2)
keyboard_char = list(string.ascii_letters + string.digits + string.punctuation + string.whitespace)

if __name__ == '__main__':
    # img = cv2.imread('keyboard.png')

    def drawhand(hands):
        if hands:
            #hand1
            hand1 = hands[0]
            lm1 = hand1["lmList"]
            box1 = hand1['bbox']
            center1 = hand1['center']
            x1 = lm1[8][0]
            y1 = lm1[8][1]
            z1 = lm1[8][2]

            x1 = center1[0]
            y1 = center1[1]
            if len(hands) == 2:
                hand2 = hands[1]
                lm2 = hand2["lmList"]
                box2 = hand2['bbox']
                center2 = hand2['center']

                x2 = center2[0]
                y2 = center2[1]
                dist = ((lm1[8][0] - lm2[8][0]) ** 2 + (lm1[8][1] - lm2[8][1]) ** 2)** 0.5
                if dist <= 200 and y1 - y2 < 110:
                    cv2.rectangle(image, (x1 + 500, y1 + 300), (x2 - 550, y2), color=(0, 255, 255), thickness=5)

    def drawcontour(cnts):
        for contour in cnts:
            if cv2.contourArea(contour) < 1000:
                continue
            cv2.drawContours(image=image,
                             contours=contour,
                             contourIdx=-1,
                             color=(0, 255, 0),
                             thickness=2,
                             lineType=cv2.LINE_AA)
            (x, y, w, h) = cv2.boundingRect(contour)
            if x + w > 600 and y + h > 400:
                cv2.rectangle(image_keyboard, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.rectangle(image_keyboard, (x + 100, y + 50), (x + w - 50, y + h - 100), (0, 255, 0), 3)

    while True:
        #variables
        ret, image = webcam.read()
        _, image_keyboard = webcam.read()
        frame_height, frame_width, _ = image.shape

        #object
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # gray = cv2.GaussianBlur(gray,(25, 25), 0)
        #
        # if first_frame is None:
        #     print(True)
        #     first_frame = gray
        #     continue
        #
        # delta_frame = cv2.absdiff(first_frame, gray)
        #
        # thresh_frame = cv2.threshold(delta_frame, 128, 255, cv2.THRESH_BINARY)[1]
        # thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)
        # thresh_frame = cv2.bitwise_not(thresh_frame)

        #color
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        sen = 80
        lower = np.array([0, 0, 255 - sen])
        upper = np.array([255, sen, 255])
        mask = cv2.inRange(hsv, lower, upper)
        # mask_invert = cv2.bitwise_not(mask)
        (cnts, _) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #contours
        drawcontour(cnts)

        #flip
        # image = cv2.flip(image, 1)

        #hand
        hands, image = detector.findHands(image, flipType=True)
        drawhand(hands)

        keyboard.drawKeyboard(image)

        #show
        cv2.imshow("image", image)
        # cv2.imshow("keyboard", image_keyboard)
        # cv2.imshow('delta', mask)

        #quit
        key = cv2.waitKey(10)
        if key == 27:
            break

    webcam.release()
    cv2.destroyAllWindows()



