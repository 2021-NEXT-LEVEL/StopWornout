from cv2 import cv2, threshold
import mediapipe as mp
from datetime import datetime
import winsound
import math
# import pygame

def main():
    cap = cv2.VideoCapture(0) 

    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils

    # pygame.mixer.init()
    # pygame.mixer.music.load("alarm.wav")

    pTime = 0 #이전 시간
    cTime = 0 #현재 시간
    frame_cnt = 0
    clap_count = 0
    cnt_flag = True
    clap_flag = False
    direction = False
    direction_current = 0
    direction_before = 0
    now = datetime.now() #현재시간
    flag0 = now.minute #현재 minute

    # clap count 어디선가 초기화
    while True:
        #minute을 계속 받아와서 매 분마다 알람을 출력하는 코드
        now = datetime.now()
        flag1 = now.minute
        
        # 제스처 인식 시, 알람 끄기
        if clap_flag == True:
            winsound.PlaySound(None, winsound.SND_ASYNC)
            clap_flag = False

        if flag0 != flag1:
            print("===========>alarm<===========")
            print("===========>alarm<===========")
            print("===========>alarm<===========")
            print("===========>alarm<===========")
            print("===========>alarm<===========")
            winsound.PlaySound('alarm.wav',winsound.SND_ASYNC)
            # pygame.mixer.music.play()
            # while pygame.mixer.music.get_busy() == True:
            #     continue
            
            flag0 = flag1
        frame_cnt += 1
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
            standard_wrist_x = 0 # 손목 x 좌표
            standard_wrist_y = 0 # 손목 y 좌표
            standard_pause_x = 0 # 중지 x 좌표
            standard_pause_y = 0 # 중지 y 좌표
            standard_distance = 0 # 손 길이 측정 -> 사용자의 거리에 따라 손 크기가 달라짐
            for handLandmarks in results.multi_hand_landmarks:
                clap_threshold = 100 # clap을 파악할 수 있는 threshold 값
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
                            standard_wrist_x = cx
                            standard_wrist_y = cy
                            if flag == 1:
                                direction_before = direction_current
                                direction_current = cx
                                # print("current: ", direction_current, "before: ", direction_before)
                                if direction_before < direction_current:
                                    direction = True
                            # print("st_wrist", standard_wrist_x, standard_wrist_y)
                        if id == 12:
                            standard_pause_x = cx
                            standard_pause_y = cy
                            # print("st_pause", standard_pause_x, standard_pause_y)
                        standard_distance = math.sqrt(pow((standard_wrist_x - standard_pause_x),2) + pow((standard_wrist_y - standard_pause_y),2))
                        # print("st_distance", standard_distance)
                #인식된 포인트에 그림 그리기
                mpDraw.draw_landmarks(img, handLandmarks, mpHands.HAND_CONNECTIONS)
            if standard_distance != 0:
                clap_threshold = standard_distance
            # print("threshold", clap_threshold)

            if frame_cnt % 3 == 0:
                # print (clap_points)
                if direction and cnt_flag and 0 < abs(clap_points[1] - clap_points[-1]) < clap_threshold :
                    cnt_flag = False
                    clap_count += 1
                    print("=========================================================clap: ", clap_count)
                elif abs(clap_points[1] - clap_points[-1]) > clap_threshold:
                    cnt_flag = True

        # if clap_count == 3:
        #     clap_count = 0
        #     clap_flag = True

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()