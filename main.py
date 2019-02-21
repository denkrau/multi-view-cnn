import sys
import getopt
from copy import deepcopy
import numpy as np
import data
import model
import globals
import cv2
import tensorflow as tf

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
			arg_ckpt = os.path.join(globals.CKPT_PATH, globals.CKPT_FILE)
		elif arg in ("-p", "--predict"):
			arg_predict = val

	#initialize objects for further use
	data = data.Data()
	model = model.Model()

	#get modelnet dataset splitted in training and testing set
	dataset = data.get_dataset(globals.DATASET_PATH, one_hot=True)

	#get weights
	weights, biases = model.get_weights()


	if arg_train == "single":
		#create placeholders for input and output data
		x = tf.placeholder(tf.float32, (None, globals.IMAGE_SIZE, globals.IMAGE_SIZE, 1), name="x")
		y = tf.placeholder(tf.float32, (None, globals.N_CLASSES), name="y")

		#rough training of model: only one image
		model.train(x, y, deepcopy(dataset), weights, biases, arg_ckpt)

	elif arg_train == "multi":
		#group module training with multiview input
		x_mv = tf.placeholder(tf.float32, (None, globals.N_VIEWS, globals.IMAGE_SIZE, globals.IMAGE_SIZE, 1), name="x_mv")
		y = tf.placeholder(tf.float32, (None, globals.N_CLASSES), name="y")

		#multi_view_dataset = copy.deepcopy(set)
		multi_view_dataset = data.single_to_multi_view(*dataset, globals.N_VIEWS)
		model.train(x_mv, y, multi_view_dataset, weights, biases, arg_ckpt)

	if arg_predict:
		img = cv2.imread(val, 0)
		if img.shape[0] != globals.IMAGE_SIZE and img.shape[1] != globals.IMAGE_SIZE:
			img = cv2.resize(img, (globals.IMAGE_SIZE, globals.IMAGE_SIZE), interpolation=cv2.INTER_AREA)
		img = np.reshape(img, [-1, globals.IMAGE_SIZE, globals.IMAGE_SIZE, 1])
		classifications = model.predict(img)
		for i in classifications:
			i = sorted(zip(globals.DATASET_LABELS, i), key=lambda x: x[1], reverse=True)
			print(*i, sep="\n")
