# mnist_recognition
��д������ʶ���ʾ���������ѡ��ͬģ��

���ߣ�Ianpanda

ʵ��������ģ�Ͷ�MNIST���ݼ���ʶ��(KNN��Softmax��CNN)

��ϵͳ������Ҫ�û�ȷ���������������Ѿ�������ɣ�
1��Python 3.5(64λ) �����ϰ汾��<br>
2��numpy 1.13.3 �����ϰ汾��<br>
3��TensorFlow 1.3.0 �����ϰ汾(������GPU�汾��������cuda v8.0��cudnn��)��<br>
4��opencv-python 3.3.1 �����ϰ汾��<br>
5��PyQt5 5.6 �����ϰ汾��<br>
6����tensorboard.exe�����ļ��м���ϵͳ����·��Path֮��(�ļ���Ϊ��*\Scripts����*�����û�python��װ·���ļ���)��<br>
ע����װ�������ʹ��pip���װ���簲װopencv-python��ʹ�����pip install opencv-python����װ��<br>

��ϵͳ�ļ�˵��
---
<div>
    <table border="0">
	  <tr>
	    <th>ģ��</th>
	    <th>�����ļ�</th>
	    <th>����˵��</th>
	  </tr>
	  <tr>
	    <td>dataload</td>
	    <td>load_MNIST_data.py</td>
	    <td>����MNIST���ݼ�</td>
	  </tr>
	  <tr>
	    <td>feature</td>
	    <td>contour.py</td>
	    <td>��ȡͼ����������</td>
	  </tr>
	  <tr>
	    <td> </td>
	    <td>correct.py</td>
	    <td>ͼ���ƫ��У��</td>
	  </tr>
	  <tr>
	    <td> </td>
	    <td>histogram.py</td>
	    <td>��ȡͼ��HOG����</td>
	  </tr>
	  <tr>
	    <td> </td>
	    <td>sharpening.py</td>
	    <td>��ȡͼ��������</td>
	  </tr>
	  <tr>
	    <td> </td>
	    <td>subsampling.py</td>
	    <td>��ȡͼ��ػ�����</td>
	  </tr>
	  <tr>
	    <td>model</td>
	    <td>MNIST_CNN.py</td>
	    <td>����CNNģ��</td>
	  </tr>
	  <tr>
	    <td>model</td>
	    <td>MNIST_KNN.py</td>
	    <td>����KNNģ��</td>
	  </tr>
	  <tr>
	    <td>model</td>
	    <td>MNIST_Softmax.py</td>
	    <td>����Softmaxģ��</td>
	  </tr>
	  <tr>
	    <td>main</td>
	    <td>mnist_GUI.py</td>
	    <td>�������壬ִ��GUI����</td>
	  </tr>
	  <tr>
	    <td>other</td>
	    <td>MNIST_datasets_decode.py</td>
	    <td>����MNIST��׼���ݼ�</td>
	  </tr>
	  <tr>
	    <td> </td>
	    <td>mnist_gui_design.py</td>
	    <td>GUI�ؼ�����</td>
	  </tr>
    </table>
</div>

