import numpy as np
import cv2
import random
import os
import globals
import tensorflow as tf

class Data(object):

	def get_dataset(self, path, one_hot=None):
		#create lists to store training and test data
		x_train = []
		y_train = []
		x_test = []
		y_test = []

		for root, dirs, files in os.walk(path):
		    #sort for alphabetical iteration
		    for file in sorted(files):
		        #print(file)
		        if file.endswith(".jpg"):
		            d = root.split("\\")
		            label = d[-2]
		            type = d[-1] # train or test
		            img = cv2.imread(os.path.join(root, file), 0)
		            if img.shape[0] != globals.IMAGE_SIZE and img.shape[1] != globals.IMAGE_SIZE:
		                img = cv2.resize(img, (globals.IMAGE_SIZE, globals.IMAGE_SIZE), interpolation=cv2.INTER_AREA)
		            if type == "train":
		                x_train.append(img)
		                y_train.append(globals.DATASET_LABELS.index(label))
		            elif type == "test":
		                x_test.append(img)
		                y_test.append(globals.DATASET_LABELS.index(label))

		#convert list to numpy array
		x_train = np.array(x_train)
		y_train = np.array(y_train)
		x_test = np.array(x_test)
		y_test = np.array(y_test)

		#one-hot encode labels
		#don't use tensorflows encoding due to evaluation in session runs
		if one_hot:
			y_test = np.eye(len(globals.DATASET_LABELS))[y_test]
			y_train = np.eye(len(globals.DATASET_LABELS))[y_train]
		x_train = x_train.reshape([-1, globals.IMAGE_SIZE, globals.IMAGE_SIZE, 1])
		x_test = x_test.reshape([-1, globals.IMAGE_SIZE, globals.IMAGE_SIZE, 1])

		return x_train, y_train, x_test, y_test

	"""Convert single views of each object to image with depth of n_views. Labels are compressed by
		taking each n_views-th element.

		Args:
			x_train: training images
			y_train: training labels
			x_test: testing images
			y_test: training labels
			n_views: number of views per object

		Returns:
			x_train: reshaped training set with views as each object's channels; size [-1, n_views, image_size, image_size, 1]
			y_train: label ob each training object
			x_test: reshaped testing set with views as each object's channels; size [-1, n_views, image_size, image_size, 1]
			y_test: label of each testing object
	"""
	def single_to_multi_view(self, x_train, y_train, x_test, y_test, n_views=globals.N_VIEWS):
		x_train = x_train.reshape([-1, n_views, globals.IMAGE_SIZE, globals.IMAGE_SIZE, 1])
		x_test = x_test.reshape([-1, n_views, globals.IMAGE_SIZE, globals.IMAGE_SIZE, 1])
		y_train = y_train[0::n_views]
		y_test = y_test[0::n_views]
		return x_train, y_train, x_test, y_test

	"""Shuffles given lists in same order

		Args:
			*lists: lists to shuffle

		Returns:
			list of each input list as tuples shuffled in same order
	"""
	def shuffle(self, *lists):
		zipped_lists = list(zip(*lists))
		random.shuffle(zipped_lists)
		# [(img1, img2, img3), (y1, y2, y3)]
		return list(zip(*zipped_lists))

	def one_hot(self, depth, *lists):
		lists_encoded = []
		for i in lists:
			lists_encoded.append(tf.one_hot(i, depth))
		return (*lists_encoded,)