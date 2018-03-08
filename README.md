# mnist_recognition
手写体数字识别的示例，界面可选不同模型

作者：Ianpanda

实现了三个模型对MNIST数据集的识别(KNN、Softmax、CNN)

本系统运行需要用户确认以下依赖环境已经配置完成：
1、Python 3.5(64位) 及以上版本；<br>
2、numpy 1.13.3 及以上版本；<br>
3、TensorFlow 1.3.0 及以上版本(若运行GPU版本，需配置cuda v8.0及cudnn库)；<br>
4、opencv-python 3.3.1 及以上版本；<br>
5、PyQt5 5.6 及以上版本；<br>
6、将tensorboard.exe所在文件夹加入系统环境路径Path之中(文件夹为“*\Scripts”，*代表用户python安装路径文件夹)。<br>
注：安装依赖库可使用pip命令安装，如安装opencv-python可使用命令“pip install opencv-python”安装。<br>

本系统文件说明
---
---
模块: dataload  feature<br><br><br><br>  model<br><br><br>  主函数<br>
包含文件:load_MNIST_data.py  contour.py  correct.py  histogram.py  sharpening.py  subsampling.py
功能说明: 载入MNIST数据集  提取图像轮廓特征  图像的偏移校正  提取图像HOG特征  提取图像锐化特征  提取图像池化特征  建立CNN模型  建立KNN模型  建立Softmax模型  解码MNIST标准数据集  主函数体，执行GUI界面  GUI控件布局
---
|模块		|包含文件			|功能说明              |
|dataload	|load_MNIST_data.py		|载入MNIST数据集	   |
|feature	|contour.py			|提取图像轮廓特征      |
|		|correct.py			|图像的偏移校正        |
|		|histogram.py			|提取图像HOG特征       |
|		|sharpening.py		|提取图像锐化特征      |
|		|subsampling.py		|提取图像池化特征      |
|model		|MNIST_CNN.py			|建立CNN模型           |
| 		|MNIST_KNN.py			|建立KNN模型           |
| 		|MNIST_Softmax.py		|建立Softmax模型       |
|		|MNIST_datasets_decode.py	|解码MNIST标准数据集   |
|主函数	|mnist_GUI.py			|主函数体，执行GUI界面 |
|		|mnist_gui_design.py	|GUI控件布局           |


