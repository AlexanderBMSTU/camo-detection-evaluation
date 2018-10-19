import numpy as np
import sys
import os
import warnings
from glob import glob
import matplotlib.pyplot as plt

caffe_root = 'C:/Projects/caffe/'  # Путь к папке со скомпилированным фреймворком
sys.path.insert(0, caffe_root + 'python')

import caffe


def vis_square(data):    
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data); plt.axis('off')
	
def classify_batch(image_paths):
    


	if os.path.isfile(caffe_root + 'models/bvlc_reference_rcnn_ilsvrc13/bvlc_reference_rcnn_ilsvrc13.caffemodel'):
		print('CaffeNet found.')

	caffe.set_mode_cpu()

	model_def = caffe_root + 'models/bvlc_reference_rcnn_ilsvrc13/deploy.prototxt'
	model_weights = caffe_root + 'models/bvlc_reference_rcnn_ilsvrc13/bvlc_reference_rcnn_ilsvrc13.caffemodel'

	net = caffe.Net(model_def,      # архитектура модели
                model_weights,  # веса модели
                caffe.TEST)     # режим работы

	mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
	mu = mu.mean(1).mean(1)  # Среднее значение каналов Imagenet

	# Преобразователь форматов для 'data'
	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

	transformer.set_transpose('data', (2,0,1))  # преобразование размерности
	transformer.set_mean('data', mu)            # задаем средние значения для каналов
	transformer.set_raw_scale('data', 255)      # масштабируем вместо [0, 1] на [0, 255]
	transformer.set_channel_swap('data', (2,1,0))  # меняем порядок каналов с RGB на BGR

	warnings.filterwarnings('ignore')

	# выгружаем лейблы ImageNet
	labels_file = caffe_root + 'data/ilsvrc12/det_synset_words.txt'
    
	labels = np.loadtxt(labels_file, str, delimiter='\t')

	net.blobs['data'].reshape(50,        # размер батча
                          3,         # 3 канала
                          227, 227)  # размер 227x227 пикселей
	
	data = np.array([np.array(caffe.io.load_image(path)) for path in image_paths])
	prob = 0
	for i in range(0,len(image_paths)):
		transformed_image = transformer.preprocess('data', data[i])
		net.blobs['data'].data[...] = transformed_image
		output = net.forward() # Распознаем 
		output_prob = output['fc-rcnn'][0]  # Вектор вероятностей для изображения
		if(output_prob.argmax() == 123): prob = prob + 1
		print("Изображение-" + str(i + 1) + " --- Распознанный класс: ", output_prob.argmax(), labels[output_prob.argmax()])
	print("Распознанно " + str(prob) + " из " + str(len(image_paths)) + " (" + str((prob/len(image_paths))*100) + "%)")