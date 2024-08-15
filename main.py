import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from time import sleep
import cvzone

cap = cv2.VideoCapture(0)
cap.set(3, 1080)
cap.set(4, 720)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

detector = HandDetector(detectionCon=0.8)
keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]]

finalText = ""


def drawAll(img, buttonList):
    overlay = img.copy()

    for button in buttonList:
        X, Y = button.pos
        button_color = (0, 128, 128)  # Teal
        text_color = (255, 255, 255)  # White
        highlight_color = (50, 205, 50)  # Lime Green

        cvzone.cornerRect(overlay, (button.pos[0], button.pos[1], button.size[0], button.size[1]), 20, rt=0)
        cv2.rectangle(overlay, button.pos, (X + button.size[0], Y + button.size[1]), button_color, cv2.FILLED)
        cv2.putText(overlay, button.text, (X + 30, Y + 50), cv2.FONT_HERSHEY_PLAIN, 2, text_color, 3)

    alpha = 0.5
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    return img


class Button:
    def __init__(self, pos, text, size=[75, 75]):
        self.pos = pos
        self.size = size
        self.text = text


buttonList = []
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        buttonList.append(Button([100 * j + 50, 100 * i + 50], key))

while True:
    success, img = cap.read()
    if not success or img is None:
        print("Error: Failed to capture image.")
        break

    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)

    hands, img = detector.findHands(img)
    if hands:
        lmList = hands[0]['lmList']
        bboxInfo = hands[0]['bbox']

    img = drawAll(img, buttonList)

    if hands:
        for button in buttonList:
            X, Y = button.pos
            w, h = button.size

            # Button colors
            button_color = (0, 128, 128)  # Teal
            text_color = (255, 255, 255)  # White
            highlight_color = (50, 205, 50)  # Lime Green

            if X < lmList[8][0] < X + w and Y < lmList[8][1] < Y + h:
                cv2.rectangle(img, button.pos, (X + w, Y + h), highlight_color, cv2.FILLED)
                cv2.putText(img, button.text, (X + 30, Y + 50), cv2.FONT_HERSHEY_PLAIN, 2, text_color, 3)

                # Slice the first two elements (x, y coordinates) to pass to findDistance
                distance, _, _ = detector.findDistance(lmList[8][:2], lmList[12][:2], img)

                # Clicking
                if distance < 80:
                    cv2.rectangle(img, button.pos, (X + w, Y + h), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, button.text, (X + 30, Y + 50), cv2.FONT_HERSHEY_PLAIN, 2, text_color, 3)
                    finalText += button.text
                    sleep(0.5)

    cv2.rectangle(img, (50, 350), (700, 450), (175, 0, 175), cv2.FILLED)
    cv2.putText(img, finalText, (60, 425), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 4)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
