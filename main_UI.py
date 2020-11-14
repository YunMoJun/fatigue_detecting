# -*- coding: utf-8 -*- 

import dlib                     # 人脸识别的库dlib
import numpy as np              # 数据处理的库numpy
import cv2                      # 图像处理的库OpenCv
import wx                       # 构造显示界面的GUI
import wx.xrc
import wx.adv
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np 
import argparse
import imutils
import datetime,time
import math
import os


COVER = './images/camera.png'
class Fatigue_detecting(wx.Frame):
    def __init__( self, parent, title ):
        wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = title, pos = wx.DefaultPosition, size = wx.Size( 873,535 ), style = wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL ) 
        self.SetSizeHints( wx.DefaultSize, wx.DefaultSize )
        self.SetBackgroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_MENU ) )
        bSizer1 = wx.BoxSizer( wx.VERTICAL )
        bSizer2 = wx.BoxSizer( wx.HORIZONTAL )
        bSizer3 = wx.BoxSizer( wx.VERTICAL )
        self.m_animCtrl1 = wx.adv.AnimationCtrl( self, wx.ID_ANY, wx.adv.NullAnimation, wx.DefaultPosition, wx.DefaultSize, wx.adv.AC_DEFAULT_STYLE ) 
        bSizer3.Add( self.m_animCtrl1, 1, wx.ALL|wx.EXPAND, 5 )        
        bSizer2.Add( bSizer3, 9, wx.EXPAND, 5 )
        bSizer4 = wx.BoxSizer( wx.VERTICAL )
        sbSizer1 = wx.StaticBoxSizer( wx.StaticBox( self, wx.ID_ANY, u"参数设置" ), wx.VERTICAL )
        sbSizer2 = wx.StaticBoxSizer( wx.StaticBox( sbSizer1.GetStaticBox(), wx.ID_ANY, u"视频源" ), wx.VERTICAL )
        gSizer1 = wx.GridSizer( 0, 2, 0, 8 )
        m_choice1Choices = [ u"摄像头ID_0", u"摄像头ID_1", u"摄像头ID_2" ]
        self.m_choice1 = wx.Choice( sbSizer2.GetStaticBox(), wx.ID_ANY, wx.DefaultPosition, wx.Size( 90,25 ), m_choice1Choices, 0 )
        self.m_choice1.SetSelection( 0 )
        gSizer1.Add( self.m_choice1, 0, wx.ALL, 5 )
        self.camera_button1 = wx.Button( sbSizer2.GetStaticBox(), wx.ID_ANY, u"开始检测", wx.DefaultPosition, wx.Size( 90,25 ), 0 )
        gSizer1.Add( self.camera_button1, 0, wx.ALL, 5 )
        self.vedio_button2 = wx.Button( sbSizer2.GetStaticBox(), wx.ID_ANY, u"打开视频文件", wx.DefaultPosition, wx.Size( 90,25 ), 0 )
        gSizer1.Add( self.vedio_button2, 0, wx.ALL, 5 )
        self.off_button3 = wx.Button( sbSizer2.GetStaticBox(), wx.ID_ANY, u"暂停", wx.DefaultPosition, wx.Size( 90,25 ), 0 )
        gSizer1.Add( self.off_button3, 0, wx.ALL, 5 )
        sbSizer2.Add( gSizer1, 1, wx.EXPAND, 5 )
        sbSizer1.Add( sbSizer2, 2, wx.EXPAND, 5 )
        sbSizer3 = wx.StaticBoxSizer( wx.StaticBox( sbSizer1.GetStaticBox(), wx.ID_ANY, u"疲劳检测" ), wx.VERTICAL )
        bSizer5 = wx.BoxSizer( wx.HORIZONTAL )
        self.yawn_checkBox1 = wx.CheckBox( sbSizer3.GetStaticBox(), wx.ID_ANY, u"打哈欠检测", wx.Point( -1,-1 ), wx.Size( -1,15 ), 0 )
        self.yawn_checkBox1.SetValue(True) 
        bSizer5.Add( self.yawn_checkBox1, 0, wx.ALL, 5 )
        self.blink_checkBox2 = wx.CheckBox( sbSizer3.GetStaticBox(), wx.ID_ANY, u"闭眼检测", wx.Point( -1,-1 ), wx.Size( -1,15 ), 0 )
        self.blink_checkBox2.SetValue(True) 
        bSizer5.Add( self.blink_checkBox2, 0, wx.ALL, 5 )
        sbSizer3.Add( bSizer5, 1, wx.EXPAND, 5 )
        bSizer6 = wx.BoxSizer( wx.HORIZONTAL )
        self.nod_checkBox7 = wx.CheckBox( sbSizer3.GetStaticBox(), wx.ID_ANY, u"点头检测", wx.Point( -1,-1 ), wx.Size( -1,15 ), 0 )
        self.nod_checkBox7.SetValue(True) 
        bSizer6.Add( self.nod_checkBox7, 0, wx.ALL, 5 )
        self.m_staticText1 = wx.StaticText( sbSizer3.GetStaticBox(), wx.ID_ANY, u"疲劳时间(秒):", wx.DefaultPosition, wx.Size( -1,15 ), 0 )
        self.m_staticText1.Wrap( -1 )
        bSizer6.Add( self.m_staticText1, 0, wx.ALL, 5 )
        m_listBox2Choices = [ u"3", u"4", u"5", u"6", u"7", u"8" ]
        self.m_listBox2 = wx.ListBox( sbSizer3.GetStaticBox(), wx.ID_ANY, wx.DefaultPosition, wx.Size( 50,24 ), m_listBox2Choices, 0 )
        bSizer6.Add( self.m_listBox2, 0, 0, 5 )
        sbSizer3.Add( bSizer6, 1, wx.EXPAND, 5 )
        sbSizer1.Add( sbSizer3, 2, 0, 5 )
        sbSizer4 = wx.StaticBoxSizer( wx.StaticBox( sbSizer1.GetStaticBox(), wx.ID_ANY, u"脱岗检测" ), wx.VERTICAL )
        bSizer8 = wx.BoxSizer( wx.HORIZONTAL )
        self.m_checkBox4 = wx.CheckBox( sbSizer4.GetStaticBox(), wx.ID_ANY, u"脱岗检测", wx.DefaultPosition, wx.Size( -1,15 ), 0 )
        self.m_checkBox4.SetValue(True)
        bSizer8.Add( self.m_checkBox4, 0, wx.ALL, 5 )
        self.m_staticText2 = wx.StaticText( sbSizer4.GetStaticBox(), wx.ID_ANY, u"脱岗时间(秒):", wx.DefaultPosition, wx.Size( -1,15 ), 0 )
        self.m_staticText2.Wrap( -1 )
        bSizer8.Add( self.m_staticText2, 0, wx.ALL, 5 )
        m_listBox21Choices = [ u"5", u"10", u"15", u"20", u"25", u"30" ]
        self.m_listBox21 = wx.ListBox( sbSizer4.GetStaticBox(), wx.ID_ANY, wx.DefaultPosition, wx.Size( 50,24 ), m_listBox21Choices, 0 )
        bSizer8.Add( self.m_listBox21, 0, 0, 5 )
        sbSizer4.Add( bSizer8, 1, 0, 5 )
        sbSizer1.Add( sbSizer4, 1, 0, 5 )
        sbSizer5 = wx.StaticBoxSizer( wx.StaticBox( sbSizer1.GetStaticBox(), wx.ID_ANY, u"分析区域" ), wx.VERTICAL )
        bSizer9 = wx.BoxSizer( wx.HORIZONTAL )
        self.m_staticText3 = wx.StaticText( sbSizer5.GetStaticBox(), wx.ID_ANY, u"检测区域：   ", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText3.Wrap( -1 )
        bSizer9.Add( self.m_staticText3, 0, wx.ALL, 5 )
        m_choice2Choices = [ u"全视频检测", u"部分区域选取" ]
        self.m_choice2 = wx.Choice( sbSizer5.GetStaticBox(), wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, m_choice2Choices, 0 )
        self.m_choice2.SetSelection( 0 )
        bSizer9.Add( self.m_choice2, 0, wx.ALL, 5 )
        sbSizer5.Add( bSizer9, 1, wx.EXPAND, 5 )
        sbSizer1.Add( sbSizer5, 1, 0, 5 )
        sbSizer6 = wx.StaticBoxSizer( wx.StaticBox( sbSizer1.GetStaticBox(), wx.ID_ANY, u"状态输出" ), wx.VERTICAL )
        self.m_textCtrl3 = wx.TextCtrl( sbSizer6.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, wx.TE_MULTILINE|wx.TE_READONLY )
        sbSizer6.Add( self.m_textCtrl3, 1, wx.ALL|wx.EXPAND, 5 )
        sbSizer1.Add( sbSizer6, 5, wx.EXPAND, 5 )
        bSizer4.Add( sbSizer1, 1, wx.EXPAND, 5 )
        bSizer2.Add( bSizer4, 3, wx.EXPAND, 5 )
        bSizer1.Add( bSizer2, 1, wx.EXPAND, 5 )

        self.SetSizer( bSizer1 )  
        self.Layout()
        self.Centre( wx.BOTH )
        self.m_choice1.Bind( wx.EVT_CHOICE, self.cameraid_choice )
        self.camera_button1.Bind( wx.EVT_BUTTON, self.camera_on )
        self.vedio_button2.Bind( wx.EVT_BUTTON, self.vedio_on )
        self.off_button3.Bind( wx.EVT_BUTTON, self.off )
        self.m_listBox2.Bind( wx.EVT_LISTBOX, self.AR_CONSEC_FRAMES )
        self.m_listBox21.Bind( wx.EVT_LISTBOX, self.OUT_AR_CONSEC_FRAMES )
        self.image_cover = wx.Image(COVER, wx.BITMAP_TYPE_ANY)
        self.bmp = wx.StaticBitmap(self.m_animCtrl1, -1, wx.Bitmap(self.image_cover))
        self.icon = wx.Icon('./images/123.ico', wx.BITMAP_TYPE_ICO)
        self.SetIcon(self.icon)
        self.Bind(wx.EVT_CLOSE, self.OnClose)     
        print("wxpython界面初始化加载完成！")
        self.VIDEO_STREAM = 0
        self.CAMERA_STYLE = False
        self.AR_CONSEC_FRAMES_check = 3
        self.OUT_AR_CONSEC_FRAMES_check = 5
        self.EYE_AR_THRESH = 0.2
        self.EYE_AR_CONSEC_FRAMES = self.AR_CONSEC_FRAMES_check
        self.MAR_THRESH = 0.5
        self.MOUTH_AR_CONSEC_FRAMES = self.AR_CONSEC_FRAMES_check
        self.HAR_THRESH = 0.3
        self.NOD_AR_CONSEC_FRAMES = self.AR_CONSEC_FRAMES_check
        self.COUNTER = 0
        self.TOTAL = 0
        self.mCOUNTER = 0
        self.mTOTAL = 0
        self.hCOUNTER = 0
        self.hTOTAL = 0
        self.oCOUNTER = 0
        self.object_pts = np.float32([[6.825897, 6.760612, 4.402142],  #33左眉左上角
                                 [1.330353, 7.122144, 6.903745],  #29左眉右角
                                 [-1.330353, 7.122144, 6.903745], #34右眉左角
                                 [-6.825897, 6.760612, 4.402142], #38右眉右上角
                                 [5.311432, 5.485328, 3.987654],  #13左眼左上角
                                 [1.789930, 5.393625, 4.413414],  #17左眼右上角
                                 [-1.789930, 5.393625, 4.413414], #25右眼左上角
                                 [-5.311432, 5.485328, 3.987654], #21右眼右上角
                                 [2.005628, 1.409845, 6.165652],  #55鼻子左上角
                                 [-2.005628, 1.409845, 6.165652], #49鼻子右上角
                                 [2.774015, -2.080775, 5.048531], #43嘴左上角
                                 [-2.774015, -2.080775, 5.048531],#39嘴右上角
                                 [0.000000, -3.116408, 6.097667], #45嘴中央下角
                                 [0.000000, -7.415691, 4.070434]])#6下巴角
        self.K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
                 0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
                 0.0, 0.0, 1.0]
        self.D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]
        self.cam_matrix = np.array(self.K).reshape(3, 3).astype(np.float32)
        self.dist_coeffs = np.array(self.D).reshape(5, 1).astype(np.float32)
        self.reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                                       [10.0, 10.0, -10.0],
                                       [10.0, -10.0, -10.0],
                                       [10.0, -10.0, 10.0],
                                       [-10.0, 10.0, 10.0],
                                       [-10.0, 10.0, -10.0],
                                       [-10.0, -10.0, -10.0],
                                       [-10.0, -10.0, 10.0]])
        self.line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
                          [4, 5], [5, 6], [6, 7], [7, 4],
                          [0, 4], [1, 5], [2, 6], [3, 7]]
        

    def __del__( self ):
        pass

    def get_head_pose(self,shape):
        image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                                shape[39], shape[42], shape[45], shape[31], shape[35],
                                shape[48], shape[54], shape[57], shape[8]])
        _, rotation_vec, translation_vec = cv2.solvePnP(self.object_pts, image_pts, self.cam_matrix, self.dist_coeffs)
        reprojectdst, _ = cv2.projectPoints(self.reprojectsrc, rotation_vec, translation_vec, self.cam_matrix,self.dist_coeffs)
        reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pose_mat = cv2.hconcat((rotation_mat, translation_vec))
        _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)
        pitch, yaw, roll = [math.radians(_) for _ in euler_angle]
        pitch = math.degrees(math.asin(math.sin(pitch)))
        roll = -math.degrees(math.asin(math.sin(roll)))
        yaw = math.degrees(math.asin(math.sin(yaw)))
        #print('pitch:{}, yaw:{}, roll:{}'.format(pitch, yaw, roll))
        return reprojectdst, euler_angle
    def eye_aspect_ratio(self,eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear
    def mouth_aspect_ratio(self,mouth):
        A = np.linalg.norm(mouth[2] - mouth[9])  # 51, 59
        B = np.linalg.norm(mouth[4] - mouth[7])  # 53, 57
        C = np.linalg.norm(mouth[0] - mouth[6])  # 49, 55
        mar = (A + B) / (2.0 * C)
        return mar


    def _learning_face(self,event):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("./model/shape_predictor_68_face_landmarks.dat")
        self.m_textCtrl3.AppendText(u"加载模型成功!!!\n")
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
        self.cap = cv2.VideoCapture(self.VIDEO_STREAM)
        if self.cap.isOpened()==True:
            self.CAMERA_STYLE = True
            self.m_textCtrl3.AppendText(u"打开摄像头成功!!!\n")
        else:
            self.m_textCtrl3.AppendText(u"摄像头打开失败!!!\n")
            self.bmp.SetBitmap(wx.Bitmap(self.image_cover))
        while(self.cap.isOpened()):
            flag, im_rd = self.cap.read()
            img_gray = cv2.cvtColor(im_rd, cv2.COLOR_RGB2GRAY)
            faces = self.detector(img_gray, 0)
            if(len(faces)!=0):
                for k, d in enumerate(faces):
                    cv2.rectangle(im_rd, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255),1)
                    shape = self.predictor(im_rd, d)
                    for i in range(68):
                        cv2.circle(im_rd, (shape.part(i).x, shape.part(i).y), 2, (0, 255, 0), -1, 8)
                    shape = face_utils.shape_to_np(shape)
                    if self.yawn_checkBox1.GetValue()== True:
                        mouth = shape[mStart:mEnd]        
                        mar = self.mouth_aspect_ratio(mouth)
                        # 使用cv2.convexHull获得凸包位置，使用drawContours画出轮廓位置进行画图操作
                        mouthHull = cv2.convexHull(mouth)
                        cv2.drawContours(im_rd, [mouthHull], -1, (0, 255, 0), 1)
                        # 同理，判断是否打哈欠    
                        if mar > self.MAR_THRESH:# 张嘴阈值0.5
                            self.mCOUNTER += 1
                        else:
                            # 如果连续3次都小于阈值，则表示打了一次哈欠
                            if self.mCOUNTER >= self.MOUTH_AR_CONSEC_FRAMES:# 阈值：3
                                self.mTOTAL += 1
                                #显示
                                cv2.putText(im_rd, "Yawning!", (10, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                self.m_textCtrl3.AppendText(time.strftime('%Y-%m-%d %H:%M ', time.localtime())+u"打哈欠\n")
                            # 重置嘴帧计数器
                            self.mCOUNTER = 0
                        cv2.putText(im_rd, "COUNTER: {}".format(self.mCOUNTER), (150, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 
                        cv2.putText(im_rd, "MAR: {:.2f}".format(mar), (300, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(im_rd, "Yawning: {}".format(self.mTOTAL), (450, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                    else:
                        pass
                    """
                    眨眼
                    """
                    if self.blink_checkBox2.GetValue()== True:
                        # 提取左眼和右眼坐标
                        leftEye = shape[lStart:lEnd]
                        rightEye = shape[rStart:rEnd]
                        # 构造函数计算左右眼的EAR值，使用平均值作为最终的EAR
                        leftEAR = self.eye_aspect_ratio(leftEye)
                        rightEAR = self.eye_aspect_ratio(rightEye)
                        ear = (leftEAR + rightEAR) / 2.0
                        leftEyeHull = cv2.convexHull(leftEye)
                        rightEyeHull = cv2.convexHull(rightEye)
                        # 使用cv2.convexHull获得凸包位置，使用drawContours画出轮廓位置进行画图操作
                        cv2.drawContours(im_rd, [leftEyeHull], -1, (0, 255, 0), 1)
                        cv2.drawContours(im_rd, [rightEyeHull], -1, (0, 255, 0), 1)
                        # 循环，满足条件的，眨眼次数+1
                        if ear < self.EYE_AR_THRESH:# 眼睛长宽比：0.2
                            self.COUNTER += 1

                        else:
                            # 如果连续3次都小于阈值，则表示进行了一次眨眼活动
                            if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:# 阈值：3
                                self.TOTAL += 1
                                self.m_textCtrl3.AppendText(time.strftime('%Y-%m-%d %H:%M ', time.localtime())+u"眨眼\n")
                            # 重置眼帧计数器
                            self.COUNTER = 0
                        # 第十四步：进行画图操作，同时使用cv2.putText将眨眼次数进行显示
                        cv2.putText(im_rd, "Faces: {}".format(len(faces)), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)     
                        cv2.putText(im_rd, "COUNTER: {}".format(self.COUNTER), (150, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 
                        cv2.putText(im_rd, "EAR: {:.2f}".format(ear), (300, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(im_rd, "Blinks: {}".format(self.TOTAL), (450, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                    else:
                        pass
                    """
                    瞌睡点头
                    """
                    if self.nod_checkBox7.GetValue()== True:
                        # 获取头部姿态
                        reprojectdst, euler_angle = self.get_head_pose(shape) 
                        har = euler_angle[0, 0]# 取pitch旋转角度
                        if har > self.HAR_THRESH:# 点头阈值0.3
                            self.hCOUNTER += 1
                        else:
                            # 如果连续3次都小于阈值，则表示瞌睡点头一次
                            if self.hCOUNTER >= self.NOD_AR_CONSEC_FRAMES:# 阈值：3
                                self.hTOTAL += 1
                                self.m_textCtrl3.AppendText(time.strftime('%Y-%m-%d %H:%M ', time.localtime())+u"瞌睡点头\n")
                            # 重置点头帧计数器
                            self.hCOUNTER = 0
                        # 绘制正方体12轴(视频流尺寸过大时，reprojectdst会超出int范围，建议压缩检测视频尺寸)
                        for start, end in self.line_pairs:
                            cv2.line(im_rd, reprojectdst[start], reprojectdst[end], (0, 0, 255))
                        # 显示角度结果
                        cv2.putText(im_rd, "X: " + "{:7.2f}".format(euler_angle[0, 0]), (10, 90), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), thickness=2)# GREEN
                        cv2.putText(im_rd, "Y: " + "{:7.2f}".format(euler_angle[1, 0]), (150, 90), cv2.FONT_HERSHEY_SIMPLEX,0.75, (255, 0, 0), thickness=2)# BLUE
                        cv2.putText(im_rd, "Z: " + "{:7.2f}".format(euler_angle[2, 0]), (300, 90), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 0, 255), thickness=2)# RED    
                        cv2.putText(im_rd, "Nod: {}".format(self.hTOTAL), (450, 90),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                    else:
                        pass
                    
                #print('嘴巴实时长宽比:{:.2f} '.format(mar)+"\t是否张嘴："+str([False,True][mar > self.MAR_THRESH]))
                #print('眼睛实时长宽比:{:.2f} '.format(ear)+"\t是否眨眼："+str([False,True][self.COUNTER>=1]))
            else:
                # 没有检测到人脸
                self.oCOUNTER+=1
                cv2.putText(im_rd, "No Face", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),3, cv2.LINE_AA)
                if self.oCOUNTER >= self.OUT_AR_CONSEC_FRAMES_check:
                    self.m_textCtrl3.AppendText(time.strftime('%Y-%m-%d %H:%M ', time.localtime())+u"员工脱岗!!!\n")
                    self.oCOUNTER = 0
                
            # 确定疲劳提示:眨眼50次，打哈欠15次，瞌睡点头30次
            if self.TOTAL >= 50 or self.mTOTAL>=15 or self.hTOTAL>=30:
                cv2.putText(im_rd, "SLEEP!!!", (100, 200),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                #self.m_textCtrl3.AppendText(u"疲劳")
                
            # opencv中imread的图片内部是BGR排序，wxPython的StaticBitmap需要的图片是RGB排序，不转换会出现颜色变换
            height,width = im_rd.shape[:2]
            image1 = cv2.cvtColor(im_rd, cv2.COLOR_BGR2RGB)
            pic = wx.Bitmap.FromBuffer(width,height,image1)
            # 显示图片在panel上：
            self.bmp.SetBitmap(pic)

        # 释放摄像头
        self.cap.release()

    def camera_on(self,event):
        """使用多线程，子线程运行后台的程序，主线程更新前台的UI，这样不会互相影响"""
        import _thread
        # 创建子线程，按钮调用这个方法，
        _thread.start_new_thread(self._learning_face, (event,))
    
    def cameraid_choice( self, event ):
        # 摄像头编号
        cameraid = int(event.GetString()[-1])# 截取最后一个字符
        if cameraid == 0:
            self.m_textCtrl3.AppendText(u"准备打开本地摄像头!!!\n")
        if cameraid == 1 or cameraid == 2:
            self.m_textCtrl3.AppendText(u"准备打开外置摄像头!!!\n")
        self.VIDEO_STREAM = cameraid
        
    def vedio_on( self, event ):  
        if self.CAMERA_STYLE == True :# 释放摄像头资源
            # 弹出关闭摄像头提示窗口
            dlg = wx.MessageDialog(None, u'确定要关闭摄像头？', u'操作提示', wx.YES_NO | wx.ICON_QUESTION)
            if(dlg.ShowModal() == wx.ID_YES):
                self.cap.release()#释放摄像头
                self.bmp.SetBitmap(wx.Bitmap(self.image_cover))#封面
                dlg.Destroy()#取消弹窗
        # 选择文件夹对话框窗口
        dialog = wx.FileDialog(self,u"选择视频检测",os.getcwd(),'',wildcard="(*.mp4)|*.mp4",style=wx.FD_OPEN | wx.FD_CHANGE_DIR)
        if dialog.ShowModal() == wx.ID_OK:
            #如果确定了选择的文件夹，将文件夹路径写到m_textCtrl3控件
            self.m_textCtrl3.SetValue(u"文件路径:"+dialog.GetPath()+"\n")
            self.VIDEO_STREAM = str(dialog.GetPath())# 更新全局变量路径
            dialog.Destroy
            """使用多线程，子线程运行后台的程序，主线程更新前台的UI，这样不会互相影响"""
            import _thread
            # 创建子线程，按钮调用这个方法，
            _thread.start_new_thread(self._learning_face, (event,))
    
    def AR_CONSEC_FRAMES( self, event ):
        self.m_textCtrl3.AppendText(u"设置疲劳间隔为:\t"+event.GetString()+"秒\n")
        self.AR_CONSEC_FRAMES_check = int(event.GetString())
        
    def OUT_AR_CONSEC_FRAMES( self, event ):
        self.m_textCtrl3.AppendText(u"设置脱岗间隔为:\t"+event.GetString()+"秒\n")
        self.OUT_AR_CONSEC_FRAMES_check = int(event.GetString())

    def off(self,event):
        """关闭摄像头，显示封面页"""
        self.cap.release()
        self.bmp.SetBitmap(wx.Bitmap(self.image_cover))
        
    def OnClose(self, evt):
        """关闭窗口事件函数"""
        dlg = wx.MessageDialog(None, u'确定要关闭本窗口？', u'操作提示', wx.YES_NO | wx.ICON_QUESTION)
        if(dlg.ShowModal() == wx.ID_YES):
            self.Destroy()
        print("检测结束，成功退出程序!!!")

            
class main_app(wx.App):
    """
     在OnInit() 里边申请Frame类，这样能保证一定是在app后调用，
     这个函数是app执行完自己的__init__函数后就会执行
    """
    # OnInit 方法在主事件循环开始前被wxPython系统调用，是wxpython独有的
    def OnInit(self):
        self.frame = Fatigue_detecting(parent=None,title="Fatigue Demo")
        self.frame.Show(True)
        return True   

    
if __name__ == "__main__":
    app = main_app()
    app.MainLoop()

