
import pygame
pygame.mixer.init()
from playsound import playsound
import math
import random
import cv2
import numpy as np
import cvzone
from cvzone.HandTrackingModule import HandDetector


cap = cv2.VideoCapture(0) # camrea
cap.set(3,1920) # width
cap.set(4,1080) # height
detector = HandDetector(detectionCon=0.7, maxHands=1)


class SankeGame:
    def __init__(self, pathFood):
        self.points = [] # تحديد جميع نقاط الثعبان
        self.length = [] # المسافة بين نقطة ونقطة
        self.currentLength = 0 # الطول الكامل للثعبان
        self.allowedLength = 100  # الطول الاجمالي المسموح به
        self.perviousHead = 0,0 # النقطة الرئسيه السابقه
        
        self.imageFood = cv2.imread(pathFood, cv2.IMREAD_UNCHANGED)
        self.imageFood = cv2.resize(self.imageFood, (0, 0), fx=0.3, fy=0.3)
        self.hFood, self.wFood, _ = self.imageFood.shape
        self.foodpoints = 0 , 0
        self.randomFoodLocation()
        
        self.snakeHead = cv2.imread('snakehead.png', cv2.IMREAD_UNCHANGED)
        self.snakeHead = cv2.resize(self.snakeHead, (70, 70))

        self.snakeBody = cv2.imread('snake-body5.png', cv2.IMREAD_UNCHANGED)
        self.snakeBody = cv2.resize(self.snakeBody, (20, 20))

    def randomFoodLocation(self): 
        self.foodpoints = random.randint(100, 600), random.randint(100, 400)   


    def update(self, imgMain , currentHead):
        px , py = self.perviousHead
        cx , cy = currentHead
        self.points.append([cx,cy])
        distance = math.hypot(cx - px , cy - py)
        self.length.append(distance)
        self.currentLength += distance
        self.perviousHead = cx , cy

        if self.currentLength > self.allowedLength:
            for i, length in enumerate(self.length):
                self.currentLength -= length
                self.length.pop(0)
                self.points.pop(0) 
                if self.currentLength < self.allowedLength:
                    break

        rx , ry = self.foodpoints
        if rx - self.wFood//2 < cx < rx + self.wFood//2 and ry - self.hFood//2 < cy < ry + self.hFood//2:
            self.randomFoodLocation() 
            self.allowedLength += 0.5    

        if self.points:
            for i in range(0, len(self.points), 1):
                bx, by = self.points[i]
                bx = int(bx - self.snakeBody.shape[1] / 2)
                by = int(by - self.snakeBody.shape[0] / 2)
                imgMain = cvzone.overlayPNG(imgMain, self.snakeBody, (bx, by))

            head_x, head_y = self.points[-1]
            hx = int(head_x - self.snakeHead.shape[1] / 2)
            hy = int(head_y - self.snakeHead.shape[0] / 2)
            imgMain = cvzone.overlayPNG(imgMain, self.snakeHead, (hx, hy))    

        rx , ry = self.foodpoints
        imgMain = cvzone.overlayPNG(imgMain, self.imageFood,(rx-self.wFood//2,ry-self.hFood//2)) 

        if len(self.points) > 20:
            pts = self.points[:-10]
            pts_array = np.array(pts, np.int32)
            pts_array = pts_array.reshape((-1, 1, 2))
            head_x, head_y = self.points[-1]

            minDist = cv2.pointPolygonTest(pts_array, (head_x, head_y), True)
            if minDist >= -1 and minDist <= 1:
                cv2.rectangle(imgMain, (300, 300), (700, 400), (255, 255, 255), cv2.FILLED)
                cv2.rectangle(imgMain, (300, 300), (700, 400), (0, 0, 0), 3)
                cv2.putText(imgMain, "Game Over", (330, 365), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
                pygame.mixer.music.load("laugh.mp3")
                pygame.mixer.music.play()
                cv2.imshow("Snake Game AI", imgMain)
                cv2.waitKey(3000)
                self.points = []
                self.length = []
                self.currentLength = 0
                self.allowedLength = 100
                self.perviousHead = 0, 0
                self.randomFoodLocation()
                      

        return imgMain

game = SankeGame('rat.png')
while True:
    success , img = cap.read()
    img = cv2.flip(img,1)
    hands, img = detector.findHands(img,flipType=False)

    if hands:
        lmList = hands[0]['lmList']
        pointIndex = lmList[8][0:2]
        #cv2.circle(img,pointIndex,20,(200,0,200),cv2.FILLED)
        img = game.update(img,pointIndex)
    cv2.imshow("Snake Game AI", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q'):
        break