import sys
import getopt
import os
import numpy as np
import cv2
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import model
import data
import globals
import matplotlib.pyplot as plt

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
	data = data.Data()
	labels = data.load_labels()

	if arg_multi is None:
		img = cv2.imread(values[0], 1 % globals.IMAGE_CHANNELS)
		if img.dtype == np.uint8:
			img = img / 255.0
		if img.shape[0] != globals.IMAGE_SIZE and img.shape[1] != globals.IMAGE_SIZE:
			img = cv2.resize(img, (globals.IMAGE_SIZE, globals.IMAGE_SIZE), interpolation=cv2.INTER_AREA)
		img = np.reshape(img, [-1, globals.IMAGE_SIZE, globals.IMAGE_SIZE, globals.IMAGE_CHANNELS])
	else:
		# airplane_0627_001.jpg
		# Format has to be name_id.jpg
		head, tail = os.path.split(values[0])
		name, ext = os.path.splitext(tail)
		name = name.split("_")
		images = []
		for i in range(arg_views):
			path = os.path.join(head, "_".join([*name[:-1], str(i).zfill(3)])) + ext
			img = cv2.imread(path, 1 % globals.IMAGE_CHANNELS)
			if img.dtype == np.uint8:
				img = img / 255.0
			if img.shape[0] != globals.IMAGE_SIZE and img.shape[1] != globals.IMAGE_SIZE:
				img = cv2.resize(img, (globals.IMAGE_SIZE, globals.IMAGE_SIZE), interpolation=cv2.INTER_AREA)
			images.append(img)
			#plt.subplot(3,4,i+1)
			#plt.imshow(img.reshape(224,224,3), vmin=0, vmax=255)
		img = np.reshape(images, [-1, arg_views, globals.IMAGE_SIZE, globals.IMAGE_SIZE, globals.IMAGE_CHANNELS])

	classifications, saliency, scores, group_ids, group_weights = model.predict(img)

	for c, s, g, w in zip(classifications, scores, group_ids, group_weights):
		c = sorted(zip(labels, c), key=lambda x: x[1], reverse=True)
		print("*** CLASSIFICATION ***")
		print(*c, sep="\n")

		if s is not None and g is not None and w is not None:
			print("\n*** GROUPING ***")
			print("View", *np.arange(12), sep="\t")
			print("=====================================================================================================")
			print("Score", *np.round(s, 3), sep="\t")
			print("Group", *g, sep="\t")

			print("\nGroup", *np.arange(12), sep="\t")
			print("=====================================================================================================")
			print("Weight", *np.around(w, 3), sep="\t")

			groups = sorted(zip(images, s, g), key=lambda x: x[2])
			_, idx = np.unique([i[2] for i in groups], return_index=True)
			for i in range(globals.N_VIEWS):
				ax = plt.subplot(1,12,i+1)
				if i in idx:
					ax.text(0.5, 1.1, round(w[groups[i][2]], 3), size=12, ha="center", transform=ax.transAxes)
				ax.text(0.5,-0.15, round(groups[i][1], 3), size=12, ha="center", transform=ax.transAxes)
				plt.xticks([])
				plt.yticks([])
				plt.imshow(groups[i][0][:,:,[2,1,0]], cmap="gray", vmin=0, vmax=1)

	if s is not None or g is not None or w is not None:
		plt.show()
