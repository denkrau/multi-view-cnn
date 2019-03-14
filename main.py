import sys
import getopt
from copy import deepcopy
import numpy as np
import data
import model
import params
import cv2
import tensorflow as tf
import os

if __name__ == "__main__":
	#first get command line arguments
	args = sys.argv[1:]
	unixOptions = "t:cp:"
	gnuOptions = ["train=", "ckpt", "predict="]
	#default params
	arg_train = ""
	arg_ckpt = None
	arg_predict = ""
	try:
		args, values = getopt.getopt(args, unixOptions, gnuOptions)
	except getopt.error as err:
		print(str(err))
		sys.exit(2)
	for arg, val in args:
		if arg in ("-t", "--train"):
			arg_train = val
		elif arg in ("-c", "--ckpt"):
			arg_ckpt = os.path.join(params.CKPT_PATH, params.CKPT_FILE)
		elif arg in ("-p", "--predict"):
			arg_predict = val

	#initialize objects for further use
	data = data.Data()
	model = model.Model()

	#get modelnet dataset splitted in training and testing set
	dataset = data.get_dataset(params.DATASET_PATH, one_hot=True)

	#get weights
	weights, biases = model.get_weights()


	if arg_train == "single":
		#create placeholders for input and output data
		x = tf.placeholder(tf.float32, (None, params.IMAGE_SIZE, params.IMAGE_SIZE, params.IMAGE_CHANNELS), name="x")
		y = tf.placeholder(tf.float32, (None, params.N_CLASSES), name="y")

		#rough training of model: only one image
		model.train(x, y, deepcopy(dataset), weights, biases, arg_ckpt)

	elif arg_train == "multi":
		#group module training with multiview input
		x_mv = tf.placeholder(tf.float32, (None, params.N_VIEWS, params.IMAGE_SIZE, params.IMAGE_SIZE, params.IMAGE_CHANNELS), name="x_mv")
		y = tf.placeholder(tf.float32, (None, params.N_CLASSES), name="y")

		#multi_view_dataset = copy.deepcopy(set)
		multi_view_dataset = data.single_to_multi_view(*dataset, params.N_VIEWS)
		model.train(x_mv, y, multi_view_dataset, weights, biases, arg_ckpt)