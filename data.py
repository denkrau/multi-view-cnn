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
		create_labels = False

		sum_labels = globals.DATASET_NUMBER_CATEGORIES + globals.DATASET_NUMBER_MATERIALS
		labels = self.load_labels()

		#supported filename is
		#path-to-dataset/category/set/category_imgId_matId_viewId.ext
		first_run = True
		for root, dirs, files in os.walk(path):
			if first_run and labels is None:
				first_run = False
				labels = dirs
				create_labels = True
			set_ = os.path.basename(root) # train or test
			#sort for alphabetical iteration
			for file in sorted(files):
				if os.path.splitext(file)[1] in globals.DATASET_FORMATS:
					label_cat = os.path.basename(os.path.dirname(root))
					label_mat_id = os.path.splitext(file)[0].split("_")[2]
					img = cv2.imread(os.path.join(root, file), 0)
					if img.shape[0] != globals.IMAGE_SIZE and img.shape[1] != globals.IMAGE_SIZE:
						img = cv2.resize(img, (globals.IMAGE_SIZE, globals.IMAGE_SIZE), interpolation=cv2.INTER_AREA)
					if set_ == "train":
						x_train.append(img)
						y_train.append([labels.index(label_cat), int(label_mat_id) + int(globals.DATASET_NUMBER_CATEGORIES)])
					elif set_ == "test":
						x_test.append(img)
						y_test.append([labels.index(label_cat), int(label_mat_id) + int(globals.DATASET_NUMBER_CATEGORIES)])
		if create_labels:
			for i in range(globals.DATASET_NUMBER_MATERIALS):
				labels.append(str(i))
			self.save_labels(labels)


		#one-hot encode labels
		#don't use tensorflows encoding due to evaluation in session runs
		if one_hot:
			y_test = [np.eye(sum_labels)[i[0]] + np.eye(sum_labels)[i[1]] for i in y_test]
			y_train = [np.eye(sum_labels)[i[0]] + np.eye(sum_labels)[i[1]] for i in y_train]
		else:
			y_train = np.array(y_train)
			y_test = np.array(y_test)

		x_train = np.reshape(x_train, [-1, globals.IMAGE_SIZE, globals.IMAGE_SIZE, 1])
		x_test = np.reshape(x_test, [-1, globals.IMAGE_SIZE, globals.IMAGE_SIZE, 1])

		return x_train, y_train, x_test, y_test

	def single_to_multi_view(self, x_train, y_train, x_test, y_test, n_views=globals.N_VIEWS):
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
		x_train = x_train.reshape([-1, n_views, globals.IMAGE_SIZE, globals.IMAGE_SIZE, 1])
		x_test = x_test.reshape([-1, n_views, globals.IMAGE_SIZE, globals.IMAGE_SIZE, 1])
		y_train = y_train[0::n_views]
		y_test = y_test[0::n_views]
		return x_train, y_train, x_test, y_test

	def shuffle(self, *lists):
		"""Shuffles given lists in same order

		Args:
			*lists: lists to shuffle

		Returns:
			list of each input list as tuples shuffled in same order
		"""
		zipped_lists = list(zip(*lists))
		random.shuffle(zipped_lists)
		# [(img1, img2, img3), (y1, y2, y3)]
		return list(zip(*zipped_lists))

	def load_labels(self):
		labels = None
		if os.path.isfile(globals.DATASET_LABELS_FILE):
			with open(globals.DATASET_LABELS_FILE, "r") as f:
				labels = [i.rstrip("\n") for i in f]
		return labels

	def save_labels(self, labels):
		with open(globals.DATASET_LABELS_FILE, "w") as f:
			for i in labels:
				f.write("%s\n" % i)