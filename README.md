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
	    <td>ģ��</td>
	    <td>�����ļ�</td>
	    <td>����˵��</td>
	  </tr>
	  <tr>
	    <th>dataload</th>
	    <th>load_MNIST_data.py</th>
	    <th>����MNIST���ݼ�</th>
	  </tr>
	  <tr>
	    <th>feature</th>
	    <th>contour.py</th>
	    <th>��ȡͼ����������</th>
	  </tr>
	  <tr>
	    <th> </th>
	    <th>correct.py</th>
	    <th>ͼ���ƫ��У��</th>
	  </tr>
	  <tr>
	    <th> </th>
	    <th>histogram.py</th>
	    <th>��ȡͼ��HOG����</th>
	  </tr>
	  <tr>
	    <th> </th>
	    <th>sharpening.py</th>
	    <th>��ȡͼ��������</th>
	  </tr>
	  <tr>
	    <th> </th>
	    <th>subsampling.py</th>
	    <th>��ȡͼ��ػ�����</th>
	  </tr>
	  <tr>
	    <th>model</th>
	    <th>MNIST_CNN.py</th>
	    <th>����CNNģ��</th>
	  </tr>
	  <tr>
	    <th>model</th>
	    <th>MNIST_KNN.py</th>
	    <th>����KNNģ��</th>
	  </tr>
	  <tr>
	    <th>model</th>
	    <th>MNIST_Softmax.py</th>
	    <th>����Softmaxģ��</th>
	  </tr>
	  <tr>
	    <th>main</th>
	    <th>mnist_GUI.py</th>
	    <th>�������壬ִ��GUI����</th>
	  </tr>
	  <tr>
	    <th>other</th>
	    <th>MNIST_datasets_decode.py</th>
	    <th>����MNIST��׼���ݼ�</th>
	  </tr>
	  <tr>
	    <th> </th>
	    <th>mnist_gui_design.py</th>
	    <th>GUI�ؼ�����</th>
	  </tr>
    </table>
</div>
|ģ��		|�����ļ�			|����˵��              |
|dataload	|load_MNIST_data.py		|����MNIST���ݼ�	   |
|feature	|contour.py			|��ȡͼ����������      |
|		|correct.py			|ͼ���ƫ��У��        |
|		|histogram.py			|��ȡͼ��HOG����       |
|		|sharpening.py		|��ȡͼ��������      |
|		|subsampling.py		|��ȡͼ��ػ�����      |
|model		|MNIST_CNN.py			|����CNNģ��           |
| 		|MNIST_KNN.py			|����KNNģ��           |
| 		|MNIST_Softmax.py		|����Softmaxģ��       |
|		|MNIST_datasets_decode.py	|����MNIST��׼���ݼ�   |
|������	|mnist_GUI.py			|�������壬ִ��GUI���� |
|		|mnist_gui_design.py	|GUI�ؼ�����           |


