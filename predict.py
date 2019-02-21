import sys
import getopt
import os
import numpy as np
import cv2
import tensorflow as tf
import model
import globals

if __name__ == "__main__":
	#first get command line arguments
	args = sys.argv[1:]
	unixOptions = "smgv:"
	gnuOptions = ["single", "multi", "groups", "views="]
	arg_single = None
	arg_multi = None
	arg_groups = None
	arg_saliency = None
	arg_views = globals.N_VIEWS
	try:
		args, values = getopt.getopt(args, unixOptions, gnuOptions)
	except getopt.error as err:
		print(str(err))
		sys.exit(2)
	for arg, val in args:
		if arg in ("-s", "--single"):
			arg_single = True
		if arg in ("-m", "--multi"):
			arg_multi = True
		if arg in ("-g", "--groups"):
			arg_groups = True
		if arg in ("-v", "--views"):
			arg_views = val

	#initialize objects for further use
	model = model.Model()

	if arg_multi is None:
		img = cv2.imread(values[0], 0)
		if img.shape[0] != globals.IMAGE_SIZE and img.shape[1] != globals.IMAGE_SIZE:
			img = cv2.resize(img, (globals.IMAGE_SIZE, globals.IMAGE_SIZE), interpolation=cv2.INTER_AREA)
		img = np.reshape(img, [-1, globals.IMAGE_SIZE, globals.IMAGE_SIZE, 1])
	else:
		# airplane_0627_001.jpg
		# Format has to be name_id.jpg
		head, tail = os.path.split(values[0])
		name, ext = os.path.splitext(tail)
		name = name.split("_")
		images = []
		for i in range(arg_views):
			path = os.path.join(head, "_".join([*name[:-1], str(i+1).zfill(3)])) + ext
			img = cv2.imread(path, 0)
			images.append(img)
		img = np.reshape(images, [-1, arg_views, globals.IMAGE_SIZE, globals.IMAGE_SIZE, 1])

	classifications, saliency, groups = model.predict(img)
	for i in classifications:
		i = sorted(zip(globals.DATASET_LABELS, i), key=lambda x: x[1], reverse=True)
		print(*i, sep="\n")
