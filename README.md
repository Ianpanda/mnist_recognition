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
<div>
    <table border="0">
	  <tr>
	    <th>模块</th>
	    <th>包含文件</th>
	    <th>功能说明</th>
	  </tr>
	  <tr>
	    <td>dataload</td>
	    <td>load_MNIST_data.py</td>
	    <td>载入MNIST数据集</td>
	  </tr>
	  <tr>
	    <td>feature</td>
	    <td>contour.py</td>
	    <td>提取图像轮廓特征</td>
	  </tr>
	  <tr>
	    <td> </td>
	    <td>correct.py</td>
	    <td>图像的偏移校正</td>
	  </tr>
	  <tr>
	    <td> </td>
	    <td>histogram.py</td>
	    <td>提取图像HOG特征</td>
	  </tr>
	  <tr>
	    <td> </td>
	    <td>sharpening.py</td>
	    <td>提取图像锐化特征</td>
	  </tr>
	  <tr>
	    <td> </td>
	    <td>subsampling.py</td>
	    <td>提取图像池化特征</td>
	  </tr>
	  <tr>
	    <td>model</td>
	    <td>MNIST_CNN.py</td>
	    <td>建立CNN模型</td>
	  </tr>
	  <tr>
	    <td></td>
	    <td>MNIST_KNN.py</td>
	    <td>建立KNN模型</td>
	  </tr>
	  <tr>
	    <td></td>
	    <td>MNIST_Softmax.py</td>
	    <td>建立Softmax模型</td>
	  </tr>
	  <tr>
	    <td>main</td>
	    <td>mnist_GUI.py</td>
	    <td>主函数体，执行GUI界面</td>
	  </tr>
	  <tr>
	    <td>other</td>
	    <td>MNIST_datasets_decode.py</td>
	    <td>解码MNIST标准数据集</td>
	  </tr>
	  <tr>
	    <td></td>
	    <td>mnist_gui_design.py</td>
	    <td>GUI控件布局</td>
	  </tr>
    </table>
</div>

