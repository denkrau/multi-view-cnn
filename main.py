import sys
import getopt
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import data
import model
import params
import cv2
import tensorflow as tf
import os

def moving_average(series, window_size):
	smoothed = []
	for i in range(len(series)):
		range_ = max(i-window_size+1, 0)
		v = series[range_:i+1]
		smoothed.append(np.mean(v))
	return smoothed

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
	dataset = data.get_dataset(params.DATASET_PATH, one_hot=True, create_labels=True)

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
		train_loss, train_accuracy, test_accuracy, learning_rate = model.train(x_mv, y, multi_view_dataset, weights, biases, arg_ckpt)

	if params.USE_PYPLOT:
		path = os.path.join(params.RESULTS_PATH, "models", os.path.basename(params.CKPT_PATH))
		if not os.path.isdir(path):
			os.makedirs(path)
		np.savez(os.path.join(path, "raw_data.npz"), loss=train_loss, training_accuracy=train_accuracy, testing_accuracy=test_accuracy)
		plt.figure(0)
		plt.xlabel("Iterations")
		plt.ylabel("Loss")
		plt.plot(train_loss, "g-", label="Loss", alpha=0.2)
		plt.plot(moving_average(train_loss, 5), "g-", label="Loss MA")
		plt.tight_layout()
		plt.savefig(os.path.join(path, "loss.png"))

		plt.figure(1)
		plt.xlabel("Iterations")
		plt.ylabel("Accuracy")
		plt.plot(train_accuracy, "g-", label="Training Accuracy", alpha=0.2)
		plt.plot(moving_average(train_accuracy, 5), "g-", label="Training Accuracy MA")
		plt.tight_layout()
		plt.savefig(os.path.join(path, "training_accuracy.png"))

		plt.figure(2)
		plt.xlabel("Iterations")
		plt.ylabel("Accuracy")
		plt.plot(test_accuracy, "g-", label="Testing Accuracy", alpha=0.2)
		plt.plot(moving_average(test_accuracy, 5), "g-", label="Testing Accuracy MA")
		plt.tight_layout()
		plt.savefig(os.path.join(path, "testing_accuracy.png"))

		plt.show()