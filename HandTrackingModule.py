import cv2
import mediapipe as mp
import numpy as np
import time


# Class to detect and track hands
class HandDetector:
    def __init__(self,
                 mode=False,
                 maxHands=2,
                 detectionConfidence=0.5,
                 trackingConfidence=0.5,
                 draw=True):

        self.draw = draw
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConfidence = detectionConfidence
        self.trackingConfidence = trackingConfidence

        # load mediapipe hand module
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionConfidence,
            min_tracking_confidence=self.trackingConfidence
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None

    # find and draw hands on the image
    def findHands(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgRGB = np.ascontiguousarray(imgRGB, dtype=np.uint8)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks and self.draw:
            for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    #get positions(x, y) of landmarks
    def findPosition(self, img, handNo=0):
        lmLists = []
        if self.results and self.results.multi_hand_landmarks:
            h, w, c = img.shape
            for handNo, handLms in enumerate(self.results.multi_hand_landmarks):
                lmList = []
                for id, lm in enumerate(handLms.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append((id, cx, cy))
                    if self.draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                lmLists.append(lmList)
        return lmLists


def main():
    pTime = 0
    fpsList = []
    cap = cv2.VideoCapture(0)  # open webcam

    width, height = 320, 240
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    detector = HandDetector(draw=False)

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        img = cv2.resize(img, (width, height))
        img = detector.findHands(img)  # detect hands
        lmLists = detector.findPosition(img)  # get landmarks

        if lmLists:
            for i, hand in enumerate(lmLists):
                print(f"Hand {i+1} - Index Finger Tip:", hand[8])

        # calculate FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime
        fpsList.append(fps)

        # auto adjust resolution if fps too low/high
        if len(fpsList) > 15:
            avgFps = sum(fpsList) / len(fpsList)
            fpsList = []
            if avgFps < 12 and width > 160:  # too slow → shrink window
                width, height = width // 2, height // 2
            elif avgFps > 25 and width < 640:  # good fps → enlarge window
                width, height = width * 2, height * 2

        cv2.putText(img, f"{int(fps)} FPS", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Video", img)

        # press ESC to quit
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
