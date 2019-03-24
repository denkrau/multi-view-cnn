import sys
import getopt
import os
import numpy as np
import cv2
import tensorflow as tf
import model
import data
import params
import matplotlib.pyplot as plt
import matplotlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
matplotlib.rcParams["savefig.directory"] = os.path.join(os.path.dirname(os.getcwd()), "Results")

if __name__ == "__main__":
	#first get command line arguments
	args = sys.argv[1:]
	unixOptions = "sgv:"
	gnuOptions = ["single", "groups", "views="]
	arg_multi = True
	arg_groups = None
	arg_saliency = None
	arg_views = params.N_VIEWS
	try:
		args, values = getopt.getopt(args, unixOptions, gnuOptions)
	except getopt.error as err:
		print(str(err))
		sys.exit(2)
	for arg, val in args:
		if arg in ("-s", "--single"):
			arg_multi = False
		if arg in ("-g", "--groups"):
			arg_groups = True
		if arg in ("-v", "--views"):
			arg_views = val

	#initialize objects for further use
	model = model.Model()
	data = data.Data()
	labels = data.load_labels()

	images = []
	if arg_multi is False:
		for val in values:
			img = cv2.imread(val, 1 % params.IMAGE_CHANNELS)
			if img.dtype == np.uint8:
				img = img / 255.0
			if img.shape[0] != params.IMAGE_SIZE and img.shape[1] != params.IMAGE_SIZE:
				img = cv2.resize(img, (params.IMAGE_SIZE, params.IMAGE_SIZE), interpolation=cv2.INTER_AREA)
			images.append(img)
		images = np.reshape(img, [-1, params.IMAGE_SIZE, params.IMAGE_SIZE, params.IMAGE_CHANNELS])
	else:
		# airplane_0627_001.jpg
		# Format has to be name_id.jpg
		for val in values:
			head, tail = os.path.split(val)
			name, ext = os.path.splitext(tail)
			name = name.split("_")
			for i in range(arg_views):
				path = os.path.join(head, "_".join([*name[:-1], str(i).zfill(3)])) + ext
				img = cv2.imread(path, 1 % params.IMAGE_CHANNELS)
				if img.dtype == np.uint8:
					img = img / 255.0
				if img.shape[0] != params.IMAGE_SIZE and img.shape[1] != params.IMAGE_SIZE:
					img = cv2.resize(img, (params.IMAGE_SIZE, params.IMAGE_SIZE), interpolation=cv2.INTER_AREA)
				images.append(img)
		images = np.reshape(images, [-1, arg_views, params.IMAGE_SIZE, params.IMAGE_SIZE, params.IMAGE_CHANNELS])

	classifications, saliency, scores, group_ids, group_weights, _ = model.predict(images)

	print(classifications.shape)

	for c, s, g, w, views in zip(classifications, scores, group_ids, group_weights, images):
		c = sorted(zip(labels if labels is not None else range(len(c)), c), key=lambda x: (x[1]), reverse=True)
		print("*** CLASSIFICATION ***")
		for i in c[:5]:
			print(i[0]+":", i[1])

		if s is not None and g is not None and w is not None:
			print("\n*** GROUPING ***")
			print("View", *np.arange(12), sep="\t")
			print("=====================================================================================================")
			print("Score", *np.round(s, 3), sep="\t")
			print("Group", *g, sep="\t")

			print("\nGroup", *np.arange(12), sep="\t")
			print("=====================================================================================================")
			print("Weight", *np.around(w, 3), "\n", sep="\t")

			groups = sorted(zip(views, s, g), key=lambda x: (x[2], x[1]))
			_, idx = np.unique([i[2] for i in groups], return_index=True)
			fig, ax = plt.subplots(1, 12, figsize=(15,2), dpi=100)
			for i in range(params.N_VIEWS):
				if i in idx:
					ax[i].text(0.5, 1.1, round(w[groups[i][2]], 3), size=12, ha="center", transform=ax[i].transAxes)
				ax[i].text(0.5,-0.15, round(groups[i][1], 3), size=12, ha="center", transform=ax[i].transAxes)
				ax[i].xaxis.set_major_locator(plt.NullLocator())
				ax[i].yaxis.set_major_locator(plt.NullLocator())
				ax[i].imshow(groups[i][0][:,:,[2,1,0]], cmap="gray", vmin=0, vmax=1)
			plt.tight_layout()
	plt.show()