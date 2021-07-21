import cv2
import mediapipe as mp

class handDetector:
    def __init__(self, mode = False, maxHands= 2, detectCon = 0.5, trackCon = 0.5) -> None:
        self.mode = mode
        self.maxHands = maxHands
        self.detectCon = detectCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw = True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)


        if self.results.multi_hand_landmarks:
            for hndLmk in self.results.multi_hand_landmarks:

                for id, lm in enumerate(hndLmk.landmark):
                    h,w,c = img.shape
                    cx, cy = int(lm.x*w), int(lm.y*h)

                # mpDraw.draw_landmarks(img, hndLmk) ### landmark points only
                if draw:
                    self.mpDraw.draw_landmarks(img, hndLmk, self.mpHands.HAND_CONNECTIONS) ### Lines connecting points in 
            return img

    def findPosition(self, img, handNo=0, draw = True, print=False):
        lmList = []
        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myhand.landmark):
                h,w,c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                if print:
                    print(id, cx, cy)
                lmList.append([id, cx,cy])
        return lmList

