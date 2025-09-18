import HandTrackingModule as htm
import cv2
import mediapipe as mp
import time

pTime = 0
fpsList = []
cap = cv2.VideoCapture(0)  # open webcam

width, height = 320, 240
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

detector = htm.HandDetector(draw=False)

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
            print(f"Hand {i + 1} - Index Finger Tip:", hand[8])

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

    # show fps on screen
    cv2.putText(img, f"{int(fps)} FPS", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Video", img)

    # press ESC to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
