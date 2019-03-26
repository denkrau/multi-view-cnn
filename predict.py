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
	unixOptions = "sgfov:c:"
	gnuOptions = ["saliency", "groups", "features", "one", "views=", "ckpt="]
	arg_multi = True
	arg_groups = True
	arg_saliency = True
	arg_views = params.N_VIEWS
	arg_ckpt = None
	arg_features = True
	try:
		args, values = getopt.getopt(args, unixOptions, gnuOptions)
	except getopt.error as err:
		print(str(err))
		sys.exit(2)
	for arg, val in args:
		if arg in ("-o", "--one"):
			arg_multi = False
		elif arg in ("-g", "--groups"):
			arg_groups = False
		elif arg in ("-v", "--views"):
			arg_views = int(val)
		elif arg in ("-c", "--ckpt"):
			arg_ckpt = val
		elif arg in ("-f", "--features"):
			arg_features = False
		elif arg in ("-s", "--saliency"):
			arg_saliency = False

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

	#predictions = classifications, saliencies, view_scores, group_ids, group_weights, correct_predictions, activations_convs
	predictions = model.predict(images, ckpt_file=arg_ckpt)

	for classification, saliency, scores, group_ids, group_weights, is_correct, activation_convs, views in zip(*predictions, images):
		classification = sorted(zip(labels if labels is not None else range(len(classification)), classification), key=lambda x: (x[1]), reverse=True)
		print("*** CLASSIFICATION ***")
		for i in classification[:5]:
			print(i[0]+":", i[1])

		if scores.any() and group_ids.any() and group_weights.any():
			print("\n*** GROUPING ***")
			print("View", *np.arange(arg_views), sep="\t")
			print("=====================================================================================================")
			print("Score", *np.round(scores, 3), sep="\t")
			print("Group", *group_ids, sep="\t")

			print("\nGroup", *np.arange(arg_views), sep="\t")
			print("=====================================================================================================")
			print("Weight", *np.around(group_weights, 3), "\n", sep="\t")

			#plots group distribution of each view
			if arg_groups:
				groups = sorted(zip(views, scores, group_ids, saliency), key=lambda x: (x[2], x[1]))
				_, idx = np.unique([i[2] for i in groups], return_index=True)
				fig, ax = plt.subplots(1, arg_views, figsize=(15,2), dpi=100)
				for i in range(arg_views):
					if i in idx:
						ax[i].text(0.5, 1.1, round(group_weights[groups[i][2]], 3), size=12, ha="center", transform=ax[i].transAxes)
					ax[i].text(0.5,-0.15, round(groups[i][1], 3), size=12, ha="center", transform=ax[i].transAxes)
					ax[i].xaxis.set_major_locator(plt.NullLocator())
					ax[i].yaxis.set_major_locator(plt.NullLocator())
					ax[i].imshow(groups[i][0][:,:,[2,1,0]], cmap="gray", vmin=0, vmax=1)
				plt.tight_layout()

		#plots saliency maps of each view
		if arg_saliency:
			if scores.any() and group_ids.any() and group_weights.any():
				groups = sorted(zip(views, scores, group_ids, saliency), key=lambda x: (x[2], x[1]))
				saliency = [s[3] for s in groups]
			fig, ax = plt.subplots(1, arg_views, figsize=(15,2), dpi=100)
			for i in range(arg_views):
				ax[i].xaxis.set_major_locator(plt.NullLocator())
				ax[i].yaxis.set_major_locator(plt.NullLocator())
				with np.errstate(divide='ignore', invalid='ignore'):
					#normalized representation
					saliency[i] = np.nan_to_num((saliency[i] - np.min(saliency[i])) / (np.max(saliency[i]) - np.min(saliency[i])))
					#dark color representation
					#saliency[i] = np.where(saliency[i] > 0, saliency[i]/np.max(saliency[i]), 0.0)
				ax[i].imshow(saliency[i][:,:,[2,1,0]], cmap="gray", vmin=0, vmax=1)
			plt.tight_layout()

		#plots feature maps of each view
		if arg_features:
			kernel_size = activation_convs.shape[-2]
			n_filter = activation_convs.shape[-1]
			grid_size = np.ceil(np.sqrt(n_filter)).astype(int)
			for v in range(params.N_VIEWS):
				fig, axes = plt.subplots(grid_size, grid_size, figsize=(15,8), dpi=100)
				fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
				axes = axes.reshape([-1])
				for i, ax in enumerate(axes):
					if i < n_filter:
						img = activation_convs[v,:,:,:,i]
						with np.errstate(divide='ignore', invalid='ignore'):
							img = np.nan_to_num((img - np.min(img)) / (np.max(img) - np.min(img)))
						#img = gaussian_filter(img, sigma=2.0)
					else:
						img = np.full([kernel_size, kernel_size], 1.0)
					ax.axis("off")
					ax.imshow(img.reshape(kernel_size, kernel_size), cmap="gray", vmin=0, vmax=1)

	#if a plot exists show it
	if arg_multi or	arg_groups or arg_saliency or arg_features:
		plt.show()