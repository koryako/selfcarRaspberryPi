
https://github.com/priya-dwivedi


Laplace 算子
http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/imgproc/imgtrans/laplace_operator/laplace_operator.html?highlight=laplace


sobel算子
http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/imgproc/imgtrans/sobel_derivatives/sobel_derivatives.html?highlight=sobel

Canny
原理 http://www.pclcn.org/study/shownews.php?lang=cn&id=111
http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html?highlight=canny#canny

http://selfdrivingcars.mit.edu/resources


三角剖分的算法比较成熟。目前有很多的库（包括命令行的和GUI的可以用）。

常用的算法叫Delaunay Triangulation，具体算法原理见 http://www.cnblogs.com/soroman/archive/2007/05/17/750430.html

这里收集一些开元的做可以测试三角剖分的库
1. Shewchuk的http://www.cs.cmu.edu/~quake/triangle.html，据说效率非常高！
2. MeshLab http://www.cs.cmu.edu/~quake/triangle.html，非常易于上手，只要新建工程，读入三维坐标点，用工具里面的Delaunay Trianglulation来可视化就好了。而且它是开源的！具体教程去网站上找吧。
3. Qhull http://www.qhull.org/
4. PCL库，http://pointclouds.org/documentation/tutorials/greedy_projection.php

无序点云快速三角化

http://www.pclcn.org/study/shownews.php?lang=cn&id=111


opencv 基本操作 https://segmentfault.com/a/1190000003742422

cv 到cv2的不同 http://www.aiuxian.com/article/p-395730.html


已经fork
https://github.com/mbeyeler/opencv-machine-learning

前言

https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/00.00-Preface.ipynb

https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/00.01-Foreword-by-Ariel-Rokem.ipynb

机器学习的味道

https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/01.00-A-Taste-of-Machine-Learning.ipynb

在OpenCV中使用数据

https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/02.00-Working-with-Data-in-OpenCV.ipynb

使用Python的NumPy软件包处理数据
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/02.01-Dealing-with-Data-Using-Python-NumPy.ipynb
在Python中加载外部数据集
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/02.02-Loading-External-Datasets-in-Python.ipynb
使用Matplotlib可视化数据
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/02.03-Visualizing-Data-Using-Matplotlib.ipynb
使用OpenCV的TrainData容器处理数据
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/02.05-Dealing-with-Data-Using-the-OpenCV-TrainData-Container-in-C%2B%2B.ipynb
监督学习的第一步

https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/03.00-First-Steps-in-Supervised-Learning.ipynb

用评分功能测量模型性能
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/03.01-Measuring-Model-Performance-with-Scoring-Functions.ipynb
了解k-NN算法
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/03.02-Understanding-the-k-NN-Algorithm.ipynb
使用回归模型预测持续成果
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/03.03-Using-Regression-Models-to-Predict-Continuous-Outcomes.ipynb
应用拉索和岭回归
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/03.04-Applying-Lasso-and-Ridge-Regression.ipynb
使用Logistic回归分类虹膜物种
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/03.05-Classifying-Iris-Species-Using-Logistic-Regression.ipynb
代表数据和工程特性

https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/04.00-Representing-Data-and-Engineering-Features.ipynb

预处理数据
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/04.01-Preprocessing-Data.ipynb
减少数据的维度
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/04.02-Reducing-the-Dimensionality-of-the-Data.ipynb
代表分类变量
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/04.03-Representing-Categorical-Variables.ipynb
表示文本特征
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/04.04-Represening-Text-Features.ipynb
代表图像
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/04.05-Representing-Images.ipynb
使用决策树进行医学诊断

https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/05.00-Using-Decision-Trees-to-Make-a-Medical-Diagnosis.ipynb

建立你的第一决策树
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/05.01-Building-Your-First-Decision-Tree.ipynb
使用决策树诊断乳腺癌
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/05.02-Using-Decision-Trees-to-Diagnose-Breast-Cancer.ipynb
使用决策树回归
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/05.03-Using-Decision-Trees-for-Regression.ipynb
用支持向量机检测行人

https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/06.00-Detecting-Pedestrians-with-Support-Vector-Machines.ipynb

实施您的第一支持向量机
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/06.01-Implementing-Your-First-Support-Vector-Machine.ipynb
检测野外行人
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/06.02-Detecting-Pedestrians-in-the-Wild.ipynb
附加SVM练习
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/06.03-Additional-SVM-Exercises.ipynb
用贝叶斯学习实现垃圾邮件过滤器

https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/07.00-Implementing-a-Spam-Filter-with-Bayesian-Learning.ipynb

实现我们的第一个贝叶斯分类器
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/07.01-Implementing-Our-First-Bayesian-Classifier.ipynb
分类电子邮件使用朴素贝叶斯
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/07.02-Classifying-Emails-Using-Naive-Bayes.ipynb
用无监督学习发现隐藏的结构

https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/08.00-Discovering-Hidden-Structures-with-Unsupervised-Learning.ipynb

了解k均值聚类
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/08.01-Understanding-k-Means-Clustering.ipynb
使用k-Means压缩彩色图像
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/08.02-Compressing-Color-Images-Using-k-Means.ipynb
使用k-Means分类手写数字
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/08.03-Classifying-Handwritten-Digits-Using-k-Means.ipynb
实施聚集层次聚类
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/08.04-Implementing-Agglomerative-Hierarchical-Clustering.ipynb
使用深度学习分类手写数字

https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/09.00-Using-Deep-Learning-to-Classify-Handwritten-Digits.ipynb

了解感知器
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/09.01-Understanding-Perceptrons.ipynb
在OpenCV中实现多层感知器
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/09.02-Implementing-a-Multi-Layer-Perceptron-in-OpenCV.ipynb
认识深度学习
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/09.03-Getting-Acquainted-with-Deep-Learning.ipynb
在OpenCV中培训MLP以分类手写数字
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/09.04-Training-an-MLP-in-OpenCV-to-Classify-Handwritten-Digits.ipynb
训练深层神经网络使用Keras分类手写数字
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/09.05-Training-a-Deep-Neural-Net-to-Classify-Handwritten-Digits-Using-Keras.ipynb
将不同的算法合并成一个合奏

https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/10.00-Combining-Different-Algorithms-Into-an-Ensemble.ipynb

了解组合方法
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/10.01-Understanding-Ensemble-Methods.ipynb
将决策树组合成随机森林
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/10.02-Combining-Decision-Trees-Into-a-Random-Forest.ipynb
使用随机森林进行人脸识别
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/10.03-Using-Random-Forests-for-Face-Recognition.ipynb
实施AdaBoost
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/10.04-Implementing-AdaBoost.ipynb
将不同的模型组合成投票分类器
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/10.05-Combining-Different-Models-Into-a-Voting-Classifier.ipynb
使用超参数调整选择正确的模型

https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/11.00-Selecting-the-Right-Model-with-Hyper-Parameter-Tuning.ipynb

评估模型
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/11.01-Evaluating-a-Model.ipynb
了解交叉验证，Bootstrapping和McNemar的测试
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/11.02-Understanding-Cross-Validation-Bootstrapping-and-McNemar's-Test.ipynb
使用网格搜索调整超参数
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/11.03-Tuning-Hyperparameters-with-Grid-Search.ipynb
链接算法一起形成管道
https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/11.04-Chaining-Algorithms-Together-to-Form-a-Pipeline.ipynb
结束语

https://github.com/mbeyeler/opencv-machine-learning/blob/master/notebooks/12.00-Wrapping-Up.ipynb
--------------
https://github.com/tensorflow/models/tree/master/object_detection
mobilenet

https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html

带有MobileNets的SSD(Single Shot Multibox Detector)

带有Inception V2的SSD

带有Resnet 101的R-FCN（Region-based Fully Convolutional Networks）

带有Resnet 101的 Faster RCNN

带有Inception Resnet v2的Faster RCNN

https://cloud.google.com/blog/big-data/2017/06/training-an-object-detector-using-cloud-machine-learning-engine



https://github.com/tensorflow/tensorflow/commit/055500bbcea60513c0160d213a10a7055f079312


mobil net 
https://github.com/tensorflow/models/tree/master/inception 准备数据
https://github.com/zehaos/MobileNet

https://github.com/balancap/SSD-Tensorflow


2017.9 

https://github.com/udacity/CarND-Term1-Starter-Kit  环境配置


http://blog.csdn.net/xukai871105/article/details/39255089 树莓派mqtt