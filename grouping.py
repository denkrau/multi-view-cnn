import numpy as np
import tensorflow as tf
import globals

class Grouping(object):
	
	def get_group_weights(self, view_scores):
		#group range and amount of views in each group
		group_size = tf.cast(tf.divide(tf.constant(1), view_scores.shape), tf.float32)
		groups = tf.histogram_fixed_width(view_scores, [0,1], tf.constant(6))

		#contains group id of each view. index of tensor corresponds to view id
		group_idx = tf.cast(tf.divide(view_scores, group_size), tf.int32)
		group_weights = []

		#create a mask for each group for weight calculation
		for i in range(0, globals.N_VIEWS):
			#create boolean mask for current group id
			mask = tf.fill([globals.N_VIEWS], i)
			mask = tf.equal(group_idx, mask)
			#calculate sum of view scores of current group
			score_sum = tf.reduce_sum(tf.boolean_mask(view_scores, mask), axis=0)
			#weight equals mean of view scores of group
			#replace NaNs with zeros
			weight = tf.cond(score_sum > 1e-7,
				lambda: tf.divide(score_sum, tf.cast(tf.count_nonzero(mask), tf.float32)),
				lambda: tf.constant(0.0))
			group_weights.append(weight)

		#normalize weights in range [0,1]
		group_weights = tf.divide(group_weights, tf.reduce_max(group_weights, axis=0))

		return group_idx, group_weights

	def get_group_descriptors(self, group_idx_and_view_descriptors):
		group_idx, view_descriptors = group_idx_and_view_descriptors
		group_descriptors = []
		#create mask for each group and calculate mean of it's view descriptors
		for i in range(0, globals.N_VIEWS):
			#create mask for current group
			mask = tf.fill([globals.N_VIEWS], i)
			mask = tf.equal(group_idx, mask)

			#calculate mean of view descriptors for current group
			descriptor_mean = tf.reduce_mean(tf.boolean_mask(view_descriptors, mask), axis=0)
			descriptor_mean = tf.where(tf.is_nan(descriptor_mean), tf.zeros_like(descriptor_mean), descriptor_mean)

			group_descriptors.append(descriptor_mean)

		return tf.convert_to_tensor(group_descriptors)


	def get_group_weights_np(self, batch_view_scores):
		"""Divides views depending on their view discrimination score into groups
	    and calculate their weights.

	    Args:
	    	batch_view_scores: discrimination score of each view; [-1, n_views]

	    Returns:
	    	batch_group_idx: index represents view id and value group id of full batch
	    	batch_avg_group_weights: normed weights of each group id of full batch
		"""
		batch_avg_group_weights_norm = []
		batch_group_idx = []
		for scores in batch_view_scores:
			# divide views related to their score into groups as histogram
			# range of histogram is [0,1]
			#print("batch_view_scores", tf.shape(batch_view_scores), batch_view_scores.shape)
			group_size = 1/scores.shape[0]
			groups = np.histogram(scores, np.arange(0, 1.0000000001, group_size), (0, 1))[0]
			#print("get_group_weights", batch_view_scores, groups)
			if np.sum(scores) == 0.0:
				print("ERROR: SCORES ARE ZERO!")
				print(batch_view_scores)
				print(groups)
			#groups = tf.histogram_fixed_width(batch_view_scores, [0, 1], batch_view_scores.shape[0].value)

			# create list of group membership for views
			# index of list represents view and value the group
			group_idx = []
			for i, s in enumerate(scores):
			    group_idx.append(int(s / group_size))
			batch_group_idx.append(group_idx)

			#create array to store discrimination scores for each group
			group_weights = np.zeros(scores.shape[0])
			for i, idx in enumerate(group_idx):
			    group_weights[idx] += scores[i]
			print(group_weights)
			#calculate mean    
			with np.errstate(divide='ignore', invalid='ignore'):
				avg_group_weights = np.nan_to_num(group_weights / groups)
			    
			#normalize the weights
			#print("get_group_weights", avg_group_weights, np.max(avg_group_weights))
			batch_avg_group_weights_norm.append(avg_group_weights / np.max(avg_group_weights))
			"""
			#TODO: should not be necessary because they have already n_view elements
			#pad list to full size, necessary that each subtensor has same length
			for i in range(globals.N_VIEWS - len(batch_group_idx)):
				batch_group_idx.append(0)
			for i in range(globals.N_VIEWS - len(batch_avg_group_weights_norm)):
				batch_avg_group_weights_norm.append(0)
			"""
		#reshape lists to match placeholders for later use
		batch_group_idx = np.reshape(batch_group_idx, [-1, globals.N_VIEWS])
		batch_avg_group_weights_norm = np.reshape(batch_avg_group_weights_norm, [-1, globals.N_VIEWS])

		return batch_group_idx, batch_avg_group_weights_norm

	def get_group_descriptors_np(self, batch_group_idx, batch_view_descriptors):
		"""Calculates the mean view descriptor of each group of each batch element, i.e. the group descriptors.

		Args:
			batch_group_idx: list of batch elements containing a list of group ids of each view; shape [-1, n_views]
			batch_view_descriptors: list of batch elements with view descriptors from convolutions; shape [-1, n_views, 1, 6, 6, 512]

		Returns:
			batch_group_descriptors: list of mean view descriptor of each group, i.e. list of group descriptors
					of every batch element; shape [-1, n_views, 1, 6, 6, 512]
		"""
		batch_group_descriptors = []
		#iterate over batches
		for group_idx, multi_view_descriptor in zip(batch_group_idx, batch_view_descriptors):
			view_descriptors = np.split(multi_view_descriptor, globals.N_VIEWS, axis=0) #[[1, 1, 6, 6, 512]]
			groups = np.unique(group_idx)
			n_groups = groups.shape[0]
			#divide each view descriptor into a group and pool them afterwards
			for i in range(globals.N_VIEWS):
				#check every group index. if group has corresponding view descriptors pool them
				#else add an ampty view descriptor to match shape of group weights vector
				if i in groups:
					single_group_descriptors = [] #[n_group_elements, 1, 6, 6, 512]
					for j, idx in enumerate(group_idx):
						if idx == i:
							single_group_descriptors.append(np.reshape(view_descriptors[j], [1, 6, 6, 512]))
					batch_group_descriptors.append(np.mean(single_group_descriptors, axis=0))
				else:
					batch_group_descriptors.append(np.zeros([1,6,6,512]))
					
		batch_group_descriptors = np.reshape(batch_group_descriptors, [-1, globals.N_VIEWS, 1, 6, 6, 512])

		return batch_group_descriptors
