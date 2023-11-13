import cv2
import mediapipe as mp 
import time


class handDetector():
    def __init__(self, mode=False, maxHands=2, modelCom=1, dectectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelCom = modelCom
        self.dectectionCon = dectectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelCom, self.dectectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0):
        lmList = list()
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
        
        return lmList
    
    def alphabet(self, img, lmList):
        if (lmList):
            if (abs(lmList[9][1] - lmList[5][1]) <= 40 and abs(lmList[13][1] - lmList[9][1]) <= 40 and abs(lmList[17][1] - lmList[13][1]) <= 40):
                if (lmList[4][2] < lmList[3][2]):
                    if (lmList[8][2] > lmList[7][2] > lmList[6][2] and lmList[12][2] > lmList[11][2] > lmList[10][2] 
                    and lmList[16][2] > lmList[15][2] > lmList[14][2] and lmList[20][2] > lmList[19][2] > lmList[18][2]):
                        if (lmList[8][1] < lmList[7][1] < lmList[6][1] and lmList[12][1] < lmList[11][1] < lmList[10][1]
                        and lmList[16][1] < lmList[15][1] < lmList[14][1] and lmList[20][1] < lmList[19][1] < lmList[18][1]): 
                            cv2.putText(img, "C", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            elif (abs(lmList[9][1] - lmList[5][1]) <= 100 and abs(lmList[13][1] - lmList[9][1]) <= 100 and abs(lmList[17][1] - lmList[13][1]) <= 100):
                if (abs(lmList[10][1] - lmList[6][1]) <= 70 and abs(lmList[14][1] - lmList[10][1]) <= 70 and abs(lmList[18][1] - lmList[14][1]) <= 70):
                    if (lmList[4][1] <= lmList[3][1]):
                        if (lmList[8][2] > lmList[7][2] >= lmList[5][2] > lmList[6][2] and lmList[12][2] > lmList[11][2] >= lmList[9][2] > lmList[10][2] 
                        and lmList[16][2] > lmList[15][2] >= lmList[13][2] > lmList[14][2] and lmList[20][2] > lmList[19][2] >= lmList[17][2] > lmList[18][2]):
                            cv2.putText(img, "A", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                    else:
                        if (lmList[8][2] < lmList[7][2] < lmList[6][2] and lmList[12][2] < lmList[11][2] < lmList[10][2] 
                        and lmList[16][2] < lmList[15][2] < lmList[14][2] and lmList[20][2] < lmList[19][2] < lmList[18][2]):
                            if (abs(lmList[10][1] - lmList[6][1]) <= 70 and abs(lmList[14][1] - lmList[10][1]) <= 70 
                            and abs(lmList[18][1] - lmList[14][1]) <= 70):
                                cv2.putText(img, "B", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                        elif (lmList[8][2] < lmList[7][2] < lmList[6][2] < lmList[5][2] and lmList[12][2] > lmList[11][2] > lmList[10][2] 
                        and lmList[16][2] > lmList[15][2] > lmList[14][2] and lmList[20][2] > lmList[19][2] > lmList[18][2]):
                            cv2.putText(img, "D", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                        elif (lmList[8][2] > lmList[7][2] > lmList[6][2] and lmList[12][2] > lmList[11][2] > lmList[10][2] 
                        and lmList[16][2] > lmList[15][2] > lmList[14][2] and lmList[20][2] > lmList[19][2] > lmList[18][2]):
                            if (lmList[8][2] < lmList[4][2] and lmList[12][2] < lmList[4][2] and lmList[16][2] < lmList[4][2] and lmList[20][2] < lmList[4][2]):
                                cv2.putText(img, "E", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

            

def main():
    pTime = 0
    cTime = 0
    detector = handDetector()

    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)

        img = detector.findHands(img)
        lmList = detector.findPosition(img)

        detector.alphabet(img, lmList)
        # if (lmList):
        #     print(lmList[8][1] < lmList[7][1] < lmList[6][1])
        #     print(lmList[12][1] < lmList[11][1] < lmList[10][1])
        #     print(lmList[16][1] < lmList[15][1] < lmList[14][1])
        #     print()
            # a = max(abs(lmList[9][1] - lmList[5][1]), abs(lmList[13][1] - lmList[9][1]))
            # a = max(a, abs(lmList[17][1] - lmList[13][1]))
            # print(a)

        
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (1200, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        
        
        cv2.imshow("Image", img)
        key_pressed = cv2.waitKey(1) & 0xFF

        if key_pressed == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows() 


if __name__ == "__main__":
    main()
