from cv2 import cv2 #그냥 import cv2 해서 에러가 나면 이렇게 해주자.
import mediapipe as mp
import time

cap = cv2.VideoCapture(0) 

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0 #이전 시간
cTime = 0 #현재 시간
cnt = 0

while True:
    cnt += 1
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # 0: 손목 / 1: 엄지4 / 2: 엄지3 / 3: 엄지2 / 4: 엄지1 / 5: 검지4 / 6: 검지3 / 7: 검지2
    # 8: 검지 1 / 9: 중지4 / 10: 중지3 / 11: 중지2 / 12: 중지1 / 13: 약지4 / 14: 약지3
    # 15: 약지2 / 16: 약지1 / 17: 새끼4 / 18: 새끼3 / 19: 새끼2 / 20: 새끼1

    #핸드를 인식하면 처리 되는 코드
    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        clap_points = {-1:0, 1:0}
        flag = 1
        for handLandmarks in results.multi_hand_landmarks:
            if len(results.multi_handedness) == 2:
                #핸드의 각 관절 포인트의 ID와 좌표를 알아 내서 원하는 그림을 그려 넣을 수 있다. 
                for id, lm in enumerate(handLandmarks.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if id == 0:
                        cv2.circle(img, (cx, cy), 10, (180, 255, 255), cv2.FILLED)
                        # print(type(handLandmarks))
                        clap_points[flag] = cx
                        flag = flag * -1
            #인식된 포인트에 그림 그리기
            mpDraw.draw_landmarks(img, handLandmarks, mpHands.HAND_CONNECTIONS)

        if cnt % 5 == 0:
            print (clap_points)
            if 0 < abs(clap_points[1] - clap_points[-1]) < 80 :
                print("clap")


    # #fps 출력
    # cTime = time.time()
    # fps = 1/(cTime-pTime)
    # pTime = cTime
    # cv2.putText(img, str(int(fps)),(10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)