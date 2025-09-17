import cv2
import mediapipe as mp
import numpy as np
import time

class HandDetector():
    def __init__(self,mode=False,maxHands=2,detectionConfidence=0.5,trackingConfidence=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConfidence = detectionConfidence
        self.trackingConfidence = trackingConfidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.detectionConfidence,self.trackingConfidence)
        self.mpDraw = mp.solutions.drawing_utils


    def findHands(self, img):

        # img = cv2.resize(img, (640, 480))
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        imgRGB = np.ascontiguousarray(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    print(id, cx, cy)
                    # if id ==0:a
                    cv2.circle(img, (int(cx), int(cy)), 10, (255, 0, 255), cv2.FILLED)

                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                # mpDraw.draw_landmarks(img, handLms)









def main():
    pTime = 0
    cTime = 0

    while True:
        success, img = cap.read()
        cap = cv2.VideoCapture(0)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),3)

        cv2.imshow('Video', img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()