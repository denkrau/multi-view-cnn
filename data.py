import numpy as np
import cv2
import random
import os
import params
import tensorflow as tf
import matplotlib.pyplot as plt

class Data(object):

	def get_dataset(self, path, train=True, test=True, one_hot=None, create_labels=False):
		#create lists to store training and test data
		x_train = []
		y_train = []
		x_test = []
		y_test = []
		has_materials = True if params.DATASET_NUMBER_MATERIALS > 0 else False
		has_categories = True if params.DATASET_NUMBER_CATEGORIES > 0 else False
		n_labels = params.get_number_labels()
		labels = self.load_labels()
		if labels is not None and len(labels) != n_labels:
			labels = None

		#supported filename is
		#path-to-dataset/category/set/category_imgId_matId_viewId.ext
		first_run = True
		for root, dirs, files in os.walk(path):
			dirs.sort()
			#on first run all categories are found as folders
			#store them as labels and append each material id each time if necessary
			if first_run and labels is None:
				first_run = False
				if params.DATASET_IS_SINGLELABEL:
					labels = []
					if has_categories and has_materials:
						#labels: A1, A2, B1, B2, C1, C2
						for d in dirs:
							for i in range(params.DATASET_NUMBER_MATERIALS):
								labels.append(d+"_"+str(i))
					elif not has_categories and has_materials:
						#labels: 1, 2, 3
						for i in range(params.DATASET_NUMBER_MATERIALS):
							labels.append(str(i))
					elif has_categories and not has_materials:
						#labels: A, B, C
						labels = dirs
					else:
						print("[ERROR] Check number of categories and materials!")
						break
				else:
					labels = dirs
					for i in range(params.DATASET_NUMBER_MATERIALS):
						labels.append(str(i))
				create_labels = True

			set_ = os.path.basename(root) # train or test
			#sort for alphabetical iteration
			for file in sorted(files):
				if os.path.splitext(file)[1] in params.DATASET_FORMATS:
					label_cat = os.path.basename(os.path.dirname(root))
					if has_materials:
						label_mat_id = os.path.splitext(file)[0].split("_")[2]
					img = cv2.imread(os.path.join(root, file), 1 % params.IMAGE_CHANNELS)
					if img.dtype == np.uint8:
						img = img.astype(np.float32)
						img = img / 255.0
					if img.shape[0] != params.IMAGE_SIZE and img.shape[1] != params.IMAGE_SIZE:
						img = cv2.resize(img, (params.IMAGE_SIZE, params.IMAGE_SIZE), interpolation=cv2.INTER_AREA)
					if set_ == "train" and train:
						x_train.append(img)
						#add label depending on number of categories and materials
						if has_categories and has_materials:
							if params.DATASET_IS_SINGLELABEL:
								y_train.append(labels.index(label_cat+"_"+label_mat_id))
							else:
								y_train.append([labels.index(label_cat), int(label_mat_id) + int(params.DATASET_NUMBER_CATEGORIES)])
						elif not has_categories and has_materials:
							y_train.append(int(label_mat_id))
						elif has_categories and not has_materials:
							y_train.append(labels.index(label_cat))
					elif set_ == "test" and test:
						x_test.append(img)
						if has_categories and has_materials:
							if params.DATASET_IS_SINGLELABEL:
								y_test.append(labels.index(label_cat+"_"+label_mat_id))
							else:
								y_test.append([labels.index(label_cat), int(label_mat_id) + int(params.DATASET_NUMBER_CATEGORIES)])
						elif not has_categories and has_materials:
							y_test.append(int(label_mat_id))
						elif has_categories and not has_materials:
							y_test.append(labels.index(label_cat))

		if create_labels:
			self.save_labels(labels)

		#one-hot encode labels
		#don't use tensorflows encoding due to evaluation in session runs
		if one_hot:
			if not params.DATASET_IS_SINGLELABEL and has_materials and has_categories:
				y_test = [np.eye(n_labels)[i[0]] + np.eye(n_labels)[i[1]] for i in y_test]
				y_train = [np.eye(n_labels)[i[0]] + np.eye(n_labels)[i[1]] for i in y_train]
				y_train = np.array(y_train, dtype=np.float32)
				y_test = np.array(y_test, dtype=np.float32)
			else:
				y_test = np.eye(n_labels, dtype=np.float32)[y_test]
				y_train = np.eye(n_labels, dtype=np.float32)[y_train]

		x_train = np.reshape(x_train, [-1, params.IMAGE_SIZE, params.IMAGE_SIZE, params.IMAGE_CHANNELS])
		x_test = np.reshape(x_test, [-1, params.IMAGE_SIZE, params.IMAGE_SIZE, params.IMAGE_CHANNELS])

		return x_train, y_train, x_test, y_test

	def single_to_multi_view(self, x_train, y_train, x_test, y_test, n_views=params.N_VIEWS):
		"""Convert single views of each object to image with depth of n_views. Labels are compressed by
		taking each n_views-th element.

		Args:
			x_train: training images
			y_train: training labels
			x_test: testing images
			y_test: training labels
			n_views: number of views per object

		Returns:
			x_train: reshaped training set with views as each object's depth; size [-1, n_views, image_size, image_size, channels]
			y_train: label ob each training object
			x_test: reshaped testing set with views as each object's depth; size [-1, n_views, image_size, image_size, channels]
			y_test: label of each testing object
		"""
		if x_train is not None:
			x_train = x_train.reshape([-1, n_views, params.IMAGE_SIZE, params.IMAGE_SIZE, params.IMAGE_CHANNELS])
		if x_test is not None:
			x_test = x_test.reshape([-1, n_views, params.IMAGE_SIZE, params.IMAGE_SIZE, params.IMAGE_CHANNELS])
		if y_train is not None:
			y_train = y_train[0::n_views]
		if y_test is not None:
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
		if os.path.isfile(params.DATASET_LABELS_FILE):
			with open(params.DATASET_LABELS_FILE, "r") as f:
				labels = [i.rstrip("\n") for i in f]
		return labels

	def save_labels(self, labels):
		with open(params.DATASET_LABELS_FILE, "w") as f:
			for i in labels:
				f.write("%s\n" % i)

	def get_amount_objects_per_category(self):
		n_objects_train = []
		n_objects_test = []
		for root, dirs, files in os.walk(params.DATASET_PATH):
			set_ = os.path.basename(root)
			if set_ == "test":
				n_objects_test.append(int(len(files)/params.N_VIEWS))
			elif set_ == "train":
				n_objects_train.append(int(len(files)/params.N_VIEWS))

		return n_objects_train, n_objects_test