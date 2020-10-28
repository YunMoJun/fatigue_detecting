# fatigue_detecting

#### 介绍
基于图像驾驶员疲劳检测技术研究

>更多学习记录参考: :tw-1f251: 
>
>**cungudafa博客：Dlib模型之驾驶员疲劳检测系列（眨眼、打哈欠、瞌睡点头、可视化界面）** 
>
>https://blog.csdn.net/cungudafa/article/details/103477960
>
>https://blog.csdn.net/cungudafa/article/details/103496881
>
>https://blog.csdn.net/cungudafa/article/details/103499230

#### 软件架构

经查阅相关文献，疲劳在人体面部表情中表现出大致三个类型：打哈欠（嘴巴张大且相对较长时间保持这一状态）、眨眼（或眼睛微闭，此时眨眼次数增多，且眨眼速度变慢）、点头（瞌睡点头）。本实验从人脸朝向、位置、瞳孔朝向、眼睛开合度、眨眼频率、瞳孔收缩率等数据入手，并通过这些数据，实时地计算出驾驶员的注意力集中程度，分析驾驶员是否疲劳驾驶和及时作出安全提示。

环境：Win10、Python3.7、anaconda3、JupyterNotebook
技术：

- Opencv：图像处理
- Dlib：一个很经典的用于图像处理的开源库，shape_predictor_68_face_landmarks.dat是一个用于人脸68个关键点检测的dat模型库，使用这个模型库可以很方便地进行人脸检测，并进行简单的应用。
- Numpy：基于Python的n维数值计算扩展。
- Imutils ：一系列使得opencv 便利的功能，包括图像旋转、缩放、平移，骨架化、边缘检测、显示
- matplotlib 图像（imutils.opencv2matplotlib(image）。
- wx：python界面工具


#### 标准参数说明

疲劳认定标准：
- 眨眼：连续3帧内，眼睛长宽比为 0.2
- 打哈欠：连续3帧内，嘴部长宽比为 0.5
- 瞌睡点头：连续3帧内，pitch（x）旋转角为 0.3

(`真实运用中需要根据不同人的眼睛大小进行检测，人的眼睛大小，俯仰头习惯都不一样，这只是一个参考值`)

![检测标准](https://images.gitee.com/uploads/images/2019/1225/234123_e66813be_5490475.png "屏幕截图.png")


#### 使用说明

一、初始化界面

![初始化功能页面](https://images.gitee.com/uploads/images/2019/1225/233300_cbfbf3c5_5490475.png "屏幕截图.png")

二、本地视频检测
1. 打开本地视频

![打开本地视频](https://images.gitee.com/uploads/images/2019/1225/233315_21291c38_5490475.png "屏幕截图.png")

2. 加载本地视频

![加载本地视频](https://images.gitee.com/uploads/images/2019/1225/233410_a607fdd9_5490475.png "屏幕截图.png")

3. 参数设置

![参数设置](https://images.gitee.com/uploads/images/2019/1225/233510_d12d6775_5490475.png "屏幕截图.png")

4. 仅闭眼检测

![仅检测闭眼](https://images.gitee.com/uploads/images/2019/1225/233748_7ce55068_5490475.png "屏幕截图.png")

5. 参数可调

![参数可调](https://images.gitee.com/uploads/images/2019/1225/233541_9343408f_5490475.png "屏幕截图.png")

三、摄像头视频流检测

![摄像头关闭提示](https://images.gitee.com/uploads/images/2019/1225/233725_7ceec090_5490475.png "屏幕截图.png")


注意：

本地视频不宜过大，会影响检测效果！



#### 参与贡献

cungudafa
