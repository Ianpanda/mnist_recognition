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
		ģ��		�����ļ�			����˵��	<br>
		dataload	load_MNIST_data.py		����MNIST���ݼ�	<br>
		feature	contour.py			��ȡͼ����������	<br>
 				correct.py			ͼ���ƫ��У��	<br>
 				histogram.py			��ȡͼ��HOG����	<br>
 				sharpening.py			��ȡͼ��������	<br>
 				subsampling.py		��ȡͼ��ػ�����	<br>
		model		MNIST_CNN.py			����CNNģ��	<br>
 				MNIST_KNN.py			����KNNģ��	<br>
 				MNIST_Softmax.py		����Softmaxģ��	<br>
				MNIST_datasets_decode.py	����MNIST��׼���ݼ�	<br>
		������		mnist_GUI.py			�������壬ִ��GUI����	<br>
				mnist_gui_design.py		GUI�ؼ�����	<br>


