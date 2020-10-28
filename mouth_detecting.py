# -*- coding: utf-8 -*-
# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np # 数据处理的库 numpy
import argparse
import imutils
import time
import dlib
import cv2
 
 
def eye_aspect_ratio(eye):
    # 垂直眼标志（X，Y）坐标
    A = dist.euclidean(eye[1], eye[5])# 计算两个集合之间的欧式距离
    B = dist.euclidean(eye[2], eye[4])
    # 计算水平之间的欧几里得距离
    # 水平眼标志（X，Y）坐标
    C = dist.euclidean(eye[0], eye[3])
    # 眼睛长宽比的计算
    ear = (A + B) / (2.0 * C)
    # 返回眼睛的长宽比
    return ear
 
def mouth_aspect_ratio(mouth):
    A = np.linalg.norm(mouth[2] - mouth[9])  # 51, 59
    B = np.linalg.norm(mouth[4] - mouth[7])  # 53, 57
    C = np.linalg.norm(mouth[0] - mouth[6])  # 49, 55
    mar = (A + B) / (2.0 * C)
    return mar


# 定义两个常数
# 眼睛长宽比
# 闪烁阈值
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 3
# 打哈欠长宽比
# 闪烁阈值
MAR_THRESH = 0.5
MOUTH_AR_CONSEC_FRAMES = 3
# 初始化帧计数器和眨眼总数
COUNTER = 0
TOTAL = 0
# 初始化帧计数器和打哈欠总数
mCOUNTER = 0
mTOTAL = 0
 
# 初始化DLIB的人脸检测器（HOG），然后创建面部标志物预测
print("[INFO] loading facial landmark predictor...")
# 第一步：使用dlib.get_frontal_face_detector() 获得脸部位置检测器
detector = dlib.get_frontal_face_detector()
# 第二步：使用dlib.shape_predictor获得脸部特征位置检测器
predictor = dlib.shape_predictor('./model/shape_predictor_68_face_landmarks.dat')
 
# 第三步：分别获取左右眼面部标志的索引
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# 第四步：打开cv2 本地摄像头
cap = cv2.VideoCapture(0)
 
# 从视频流循环帧
while True:
    # 第五步：进行循环，读取图片，并对图片做维度扩大，并进灰度化
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=720)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 第六步：使用detector(gray, 0) 进行脸部位置检测
    rects = detector(gray, 0)
    
    # 第七步：循环脸部位置信息，使用predictor(gray, rect)获得脸部特征位置的信息
    for rect in rects:
        shape = predictor(gray, rect)
        
        # 第八步：将脸部特征信息转换为数组array的格式
        shape = face_utils.shape_to_np(shape)
        
        # 第九步：提取左眼和右眼坐标
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        # 嘴巴坐标
        mouth = shape[mStart:mEnd]
        
        
        # 第十步：构造函数计算左右眼的EAR值，使用平均值作为最终的EAR
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        # 打哈欠
        mar = mouth_aspect_ratio(mouth)
 
        # 第十一步：使用cv2.convexHull获得凸包位置，使用drawContours画出轮廓位置进行画图操作
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
 
        # 第十二步：进行画图操作，用矩形框标注人脸
        left = rect.left()
        top = rect.top()
        right = rect.right()
        bottom = rect.bottom()
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)    
 
        '''
            分别计算左眼和右眼的评分求平均作为最终的评分，如果小于阈值，则加1，如果连续3次都小于阈值，则表示进行了一次眨眼活动
        '''
        # 第十三步：循环，满足条件的，眨眼次数+1
        if ear < EYE_AR_THRESH:# 眼睛长宽比：0.2
            COUNTER += 1
           
        else:
            # 如果连续3次都小于阈值，则表示进行了一次眨眼活动
            if COUNTER >= EYE_AR_CONSEC_FRAMES:# 阈值：3
                TOTAL += 1
            # 重置眼帧计数器
            COUNTER = 0
            
        # 第十四步：进行画图操作，同时使用cv2.putText将眨眼次数进行显示
        cv2.putText(frame, "Faces: {}".format(len(rects)), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Blinks: {}".format(TOTAL), (150, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "COUNTER: {}".format(COUNTER), (300, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (450, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        '''
            计算张嘴评分，如果小于阈值，则加1，如果连续3次都小于阈值，则表示打了一次哈欠，同一次哈欠大约在3帧
        '''
        # 同理，判断是否打哈欠    
        if mar > MAR_THRESH:# 张嘴阈值0.5
            mCOUNTER += 1
            cv2.putText(frame, "Yawning!", (10, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # 如果连续3次都小于阈值，则表示打了一次哈欠
            if mCOUNTER >= MOUTH_AR_CONSEC_FRAMES:# 阈值：3
                mTOTAL += 1
            # 重置嘴帧计数器
            mCOUNTER = 0
        cv2.putText(frame, "Yawning: {}".format(mTOTAL), (150, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "mCOUNTER: {}".format(mCOUNTER), (300, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 
        cv2.putText(frame, "MAR: {:.2f}".format(mar), (480, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        # 第十五步：进行画图操作，68个特征点标识
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    print('嘴巴实时长宽比:{:.2f} '.format(mar)+"\t是否张嘴："+str([False,True][mar > MAR_THRESH]))
    print('眼睛实时长宽比:{:.2f} '.format(ear)+"\t是否眨眼："+str([False,True][COUNTER>=1]))
    
    # 确定疲劳提示
    if TOTAL >= 50 or mTOTAL>=15:
        cv2.putText(frame, "SLEEP!!!", (100, 200),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
    # 按q退出
    cv2.putText(frame, "Press 'q': Quit", (20, 500),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (84, 255, 159), 2)
    # 窗口显示 show with opencv
    cv2.imshow("Frame", frame)
    
    # if the `q` key was pressed, break from the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# 释放摄像头 release camera
cap.release()
# do a bit of cleanup
cv2.destroyAllWindows()
