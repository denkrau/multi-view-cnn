import os
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import grouping
import params
import data

class Model(object):

	def __init__(self):
		self.grouping = grouping.Grouping()
		self.data = data.Data()

	def cnn_convs(self, x, weights, biases):
		"""Apply five convolutions and three pooling operations to input 
		according to vgg-m architecture.

		Args:
			x: input image
			weights: weights of filter
			biases: biases of layers

		Returns:
			conv5: output of fifth convolutional layer; shape [1, 6, 6, 512]
		"""
		with tf.name_scope("conv1"):
			conv1 = self.conv2d(x, weights['wc1'], biases['bc1'], strides=2, padding="VALID") #(1, 109, 109, 96)
			conv1 = self.max_pool2d(conv1, size=3, strides=2, padding="VALID") #(1, 54, 54, 96)
		with tf.name_scope("conv2"):
			conv2 = self.conv2d(conv1, weights['wc2'], biases['bc2'], strides=2, padding="SAME") #(1, 27, 27, 256)
			conv2 = self.max_pool2d(conv2, size=3, strides=2, padding="VALID") #(1, 13, 13, 256)
		with tf.name_scope("conv3"):
			conv3 = self.conv2d(conv2, weights['wc3'], biases['bc3'], strides=1, padding="SAME") #(1, 13, 13, 512)
			#conv3 = max_pool2d(conv3, size=3) #
		with tf.name_scope("conv4"):
			conv4 = self.conv2d(conv3, weights['wc4'], biases['bc4'], strides=1, padding="SAME") #(1, 13, 13, 512)
			#conv4 = max_pool2d(conv4, size=3) #
		with tf.name_scope("conv5"):
			conv5 = self.conv2d(conv4, weights['wc5'], biases['bc5'], strides=1, padding="SAME") #(1, 13, 13, 512)
			conv5 = self.max_pool2d(conv5, size=3, strides=2, padding="VALID") #(1, 6, 6, 512)
		return conv5

	def cnn_fcs(self, x, weights, biases):
		"""Applies three fully connected layer operations to input
		according to vgg-m architecture.

		Args:
			x: input (output of conv5)
			weights: weights of filter
			biases: biases of layers

		Returns:
			fc3: input membership to classes, i.e. predicted classes
		"""
		with tf.name_scope("dropout"):
			dropout_prob = tf.placeholder_with_default(0.0, ())
		with tf.name_scope("fc1"):
			fc1 = tf.reshape(x, [-1, weights['wd1'].get_shape().as_list()[0]])
			fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
			fc1 = tf.nn.relu(fc1) #(1, 4096)
			fc1 = tf.nn.dropout(fc1, keep_prob=1-dropout_prob)
		with tf.name_scope("fc2"):
			fc2 = tf.reshape(fc1, [-1, weights['wd2'].get_shape().as_list()[0]])
			fc2 = tf.add(tf.matmul(fc2, weights['wd2']), biases['bd2'])
			fc2 = tf.nn.relu(fc2) #(1, 4096)
			fc2 = tf.nn.dropout(fc2, keep_prob=1-dropout_prob)
		with tf.name_scope("fc3"):
			#fc3 = tf.reshape(fc2, [-1, weights['wd3'].get_shape().as_list()[0]])
			fc3 = tf.add(tf.matmul(fc2, weights['wd3']), biases['bd3']) #(1,n_classes)

		return fc3

	def cnn_fcs_grouping(self, group_descriptors, group_weights, weights, biases):
		"""Classifies each batch element object from views by taking group descriptors of whole batch as input.

		Args:
			group_descriptors: group descriptors of pooled views, for simplicity
							   they are padded with zero descriptors to match amount
							   of views; shape [-1, n_views, 1, 6, 6, 512]
			group_weights: weight of each group; shape [-1, views]
			weights: model weights
			biases: model biases

		Returns:
			fc3: predicted classification of object or views, respectively; shape [-1, 1, n_classes]
		"""
		shape_descriptors = tf.map_fn(self.get_shape_descriptors, group_descriptors)
		shape_descriptors = tf.reshape(shape_descriptors, [-1, params.N_VIEWS, weights["wd2"].get_shape().as_list()[0]])

		#make tensors ready for single shape descriptor calculation
		#transpose, but keep first dimension as batch dimension
		shape_descriptors = tf.transpose(shape_descriptors, perm=[0,2,1])

		#append dimension at the end for valid matrix multiplication
		group_weights = tf.expand_dims(group_weights, axis=2)

		# compute shape descriptor dependent on group weights
		final_shape_descriptor = tf.divide(tf.matmul(shape_descriptors, group_weights), tf.expand_dims(tf.reduce_sum(group_weights, 1), axis=2))

		#transpose again for final matrix multiplication
		final_shape_descriptor = tf.transpose(final_shape_descriptor, perm=[0,2,1]) #[-1, 1, 4096]

		final_shape_descriptor = tf.reshape(final_shape_descriptor, [-1, weights['wd3'].get_shape().as_list()[0]]) #[-1, 4096]
		fc3 = tf.add(tf.matmul(final_shape_descriptor, weights['wd3']), biases['bd3'])
		
		return fc3

	def cnn_grouping(self, x, weights, biases):
		"""Calculates the view discrimination score of a view.

		Args:
			x: view descriptor as input; shape [1, 6, 6, 512]
			weights: weights of cnn
			biases: biases of cnn

		Returns:
			fc1: view discrimination score of input view
		"""
		with tf.name_scope("fc4"):
			fc1 = tf.reshape(x, [-1, weights['wd4'].get_shape().as_list()[0]])
			fc1 = tf.add(tf.matmul(fc1, weights['wd4']), biases['bd4'])
			fc1 = tf.nn.leaky_relu(fc1)
			#fc1 = tf.nn.dropout(fc1, 0.5)
			#add constant to avoid log(0) and therefore NaNs in cost function
			fc1 = tf.sigmoid(tf.log(tf.abs(fc1)+tf.constant(0.000001)), name="view_discr_score")
		return fc1

	def train(self, x, y, dataset, weights, biases, ckpt=None, find_lr=None):
		"""Trains the network with given training and testing images and labels

		Args:
			x: input placeholder of shape [-1, image_height, image_width, 1] for single view or
			   [-1, n_views, image_height, image_width, 1] for multi-view
			y: label placeholder of shape [-1, n_classes]
			dataset: tuple with training images and labels and testing images and labels
			weights: dictionary with model weights
			biases: dictionary with model biases
			ckpt: checkpoint file path for pre-trained model params
		"""
		x_train, y_train, x_test, y_test = dataset
		is_multi_view = False
		with tf.name_scope("dropout"):
			dropout_prob = tf.placeholder_with_default(0.0, ())
		# if shape of x placeholder relates to n_views dimension a multi-view input is supposed
		if x.shape[1] == params.N_VIEWS:
			is_multi_view = True
			batch_size = params.BATCH_SIZE_MULTI
			# feed each batch element of input in first part of cnn (conv layers) to get view descriptors and scores of each channel, i.e. view
			# view_descriptors = [-1, n_views, 1, 6, 6, 512], view_discrimination_scores = [-1, n_views, 1, 1] -> [-1, n_views]
			with tf.name_scope("view_descriptors_and_scores"):
				view_descriptors, view_discrimination_scores = tf.map_fn(self.get_view_descriptors_and_scores, x, dtype=(tf.float32, tf.float32))
				if params.USE_SUMMARY:
					tf.summary.histogram("view_descriptors", view_descriptors)
					tf.summary.histogram("view_scores", view_discrimination_scores)
				#make scores more compact
				view_discrimination_scores = tf.reshape(view_discrimination_scores, [-1, params.N_VIEWS])

			with tf.name_scope("group_weights"):
				batch_group_idx, batch_group_weights = tf.map_fn(self.grouping.get_group_weights, view_discrimination_scores, dtype=(tf.int32, tf.float32))

			with tf.name_scope("grouping"):
				batch_group_descriptors = tf.map_fn(self.grouping.get_group_descriptors, [batch_group_idx, view_descriptors], dtype=tf.float32)
				pred = self.cnn_fcs_grouping(batch_group_descriptors, batch_group_weights, weights, biases)
		else:
			batch_size = params.BATCH_SIZE_SINGLE
			# feed input to connected cnn
			pred = self.cnn_convs(x, weights, biases)
			pred = self.cnn_fcs(pred, weights, biases)

		#sigmoid for multi-label, softmax for single-label
		if params.DATASET_IS_SINGLELABEL:
			cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
		else:
			cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y))
		#cost = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y), axis=1))
		#learning_rate = tf.placeholder(tf.float32, ())
		learning_rate = params.LEARNING_RATE
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
		if params.DATASET_IS_SINGLELABEL:
			correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
		else:
			correct_prediction = tf.equal(tf.round(tf.sigmoid(pred)), y)		
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		init = tf.global_variables_initializer()
		saver = tf.train.Saver()
		if params.USE_SUMMARY:
			merged = tf.summary.merge_all()
		#config = tf.ConfigProto()
		#config.gpu_options.allow_growth = True
		with tf.Session() as sess:
			#lr = self.get_learning_rate(find_lr)
			if ckpt is None:
				sess.run(init)
			else:
				saver.restore(sess, ckpt)
			if params.USE_SUMMARY:
				summary_writer = tf.summary.FileWriter(params.SUMMARY_PATH, sess.graph)

			train_loss = []
			learning_rates = []

			for i in range(params.TRAINING_EPOCHS if find_lr is None else 100):
				x_train, y_train = self.data.shuffle(x_train, y_train)
				x_test, y_test = self.data.shuffle(x_test, y_test)
				
				if find_lr is None:
					#reset lists due to performance reasons
					train_loss = []
					learning_rates = []
				test_loss = []
				train_accuracy = []
				test_accuracy = []

				for batch in range(len(x_train)//batch_size):
					print("Epoch", i, "Batch", batch, "of", len(x_train)//batch_size, end="\r")
					batch_x = x_train[batch*batch_size:min((batch+1)*batch_size,len(x_train))]
					batch_y = y_train[batch*batch_size:min((batch+1)*batch_size,len(y_train))]
					#learning_rates.append(lr)
					# Run optimization op (backprop).
					    # Calculate batch loss and accuracy

					if is_multi_view:
						if params.USE_SUMMARY:
							#summary, opt, scores, descr = sess.run([merged, optimizer, view_discrimination_scores, view_descriptors], feed_dict={x: batch_x, y: batch_y, learning_rate: lr})
							summary, opt, scores, descr = sess.run([merged, optimizer, view_discrimination_scores, view_descriptors], feed_dict={x: batch_x, y: batch_y, dropout_prob: params.DROPOUT_PROB})
						else:
							#opt = sess.run([optimizer], feed_dict={x: batch_x, y: batch_y, learning_rate: lr})
							opt, s = sess.run([optimizer,view_discrimination_scores], feed_dict={x: batch_x, y: batch_y, dropout_prob: params.DROPOUT_PROB})
						#loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, learning_rate: lr})
						loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})
						if params.USE_SUMMARY:
							summary_writer.add_summary(summary)

					else:
						#opt = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, learning_rate: lr})
						opt = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, dropout_prob: params.DROPOUT_PROB})
						#loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, learning_rate: lr})
						loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})

					train_loss.append(loss)

					#finding learning rate is active, therefore break if latest loss is much larger than loss before
					if find_lr is not None and i > 0:
						if train_loss[-1] > 4 * train_loss[-2]:
							return train_loss, learning_rates, train_accuracy, test_accuracy

					#lr = self.update_learning_rate(lr, find_lr)
				print(s)
				print("Epoch " + str(i) + ", Loss= " + \
				          "{:.6f}".format(train_loss[-1]) + ", Training Accuracy= " + \
				          "{:.5f}".format(acc))
				#print("Optimization Finished!")

				# Calculate accuracy for all images
				#divide in batches and calculate mean to prevent OOM error
				if find_lr is None:
					test_batch_accuracy = []
					test_batch_valid_loss = []
					for test_batch in range(len(x_test)//batch_size):
						batch_x_test = x_test[test_batch*batch_size:min((test_batch+1)*batch_size, len(x_test))]
						batch_y_test = y_test[test_batch*batch_size:min((test_batch+1)*batch_size, len(y_test))]
						if is_multi_view:
							test_acc, valid_loss = sess.run([accuracy,cost], feed_dict={x: batch_x_test, y : batch_y_test})
						else:
							test_acc, valid_loss = sess.run([accuracy,cost], feed_dict={x: batch_x_test, y : batch_y_test})
						test_batch_accuracy.append(test_acc)
						test_batch_valid_loss.append(valid_loss)

					test_loss.append(np.mean(test_batch_valid_loss))
					test_accuracy.append(np.mean(test_batch_accuracy))

					print("Testing Accuracy:","{:.5f}".format(test_accuracy[-1]))

				train_accuracy.append(acc)
			try:
				saver.save(sess, os.path.join(params.CKPT_PATH, params.CKPT_FILE))
			except Exception as e:
				print(e)
			print("Training finished!")
			if params.USE_SUMMARY:
				summary_writer.close()

		return train_loss, learning_rates, train_accuracy, test_accuracy

	def predict(self, img, labels=None, n_views=params.N_VIEWS, ckpt_file=None):
		weights, biases = self.get_weights()
		is_multi_view = None
		saliencies = []
		groups = []
		view_scores = []
		group_ids = []
		group_weights = []
		correct_predictions = []
		classifications = []
		
		#if multi view object is given
		if img.shape[1] == n_views:
			is_multi_view = True
			batch_size = params.BATCH_SIZE_MULTI
			x = tf.placeholder(tf.float32, [None, n_views, params.IMAGE_SIZE, params.IMAGE_SIZE, params.IMAGE_CHANNELS], name="x")

			with tf.name_scope("view_descriptors_and_scores"):
				view_descriptors, view_discrimination_scores = tf.map_fn(self.get_view_descriptors_and_scores, x, dtype=(tf.float32, tf.float32))
				view_discrimination_scores = tf.reshape(view_discrimination_scores, [-1, params.N_VIEWS])

			with tf.name_scope("group_weights"):
				batch_group_idx, batch_group_weights = tf.map_fn(self.grouping.get_group_weights, view_discrimination_scores, dtype=(tf.int32, tf.float32))

			with tf.name_scope("grouping"):
				batch_group_descriptors = tf.map_fn(self.grouping.get_group_descriptors, [batch_group_idx, view_descriptors], dtype=tf.float32)
				pred = self.cnn_fcs_grouping(batch_group_descriptors, batch_group_weights, weights, biases)
				if params.DATASET_IS_SINGLELABEL:
					pred = tf.nn.softmax(pred)
				else:
					pred = tf.nn.sigmoid(pred)
		else:
			is_multi_view = False
			batch_size = params.BATCH_SIZE_SINGLE
			# feed input to connected cnn
			x = tf.placeholder(tf.float32, [None, params.IMAGE_SIZE, params.IMAGE_SIZE, params.IMAGE_CHANNELS], name="x")
			pred = self.cnn_convs(x, weights, biases)
			pred = self.cnn_fcs(pred, weights, biases)
			pred = tf.nn.softmax(pred)
			saliency = tf.gradients(pred, x)

		y = tf.placeholder(tf.float32, [None, params.N_CLASSES])
		correct_prediction = None
		if labels is not None:
			if params.DATASET_IS_SINGLELABEL:
				correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
			else:
				correct_prediction = tf.equal(tf.round(tf.sigmoid(pred)), y)

		saver = tf.train.Saver()
		if ckpt_file is None:
			ckpt_file = os.path.join(params.CKPT_PATH, params.CKPT_FILE)

		with tf.Session() as sess:
			saver.restore(sess, ckpt_file)

			for batch in range(max(img.shape[0]//batch_size, 1)):
				batch_x = img[batch*batch_size:min((batch+1)*batch_size,img.shape[0])]
				classification = sess.run(pred, feed_dict={x: batch_x})
				classifications.append(classification)

				if labels is not None:
					batch_y = labels[batch*batch_size:min((batch+1)*batch_size,labels.shape[0])]
					is_correct = sess.run(correct_prediction, feed_dict={x: batch_x, y: batch_y})
					correct_predictions.append(is_correct)

				if is_multi_view:
					g_ids, g_weights, v_scores = sess.run([batch_group_idx, batch_group_weights, view_discrimination_scores], feed_dict={x: batch_x})
					group_ids.append(g_ids)
					group_weights.append(g_weights)
					view_scores.append(v_scores)
				else:
					saliency = sess.run(saliency, feed_dict={x: batch_x})
					saliencies.append(saliency)
			
			saliencies = np.reshape(saliencies, [-1, params.IMAGE_SIZE, params.IMAGE_SIZE])
			groups =  np.reshape(groups, [-1, params.N_VIEWS])
			view_scores =  np.reshape(view_scores, [-1, params.N_VIEWS])
			group_ids =  np.reshape(group_ids, [-1, params.N_VIEWS])
			group_weights =  np.reshape(group_weights, [-1, params.N_VIEWS])
			correct_predictions =  np.reshape(correct_predictions, [-1])
			classifications =  np.reshape(classifications, [-1, params.N_CLASSES])
			
		return classifications, saliencies, view_scores, group_ids, group_weights, correct_predictions

	def get_weights(self):
		with tf.variable_scope("cnn", reuse=tf.AUTO_REUSE):
			with tf.name_scope("weights"):
				weights = {
					#shape = (filter_size_row, filter_size_col, channels of input, number of convs)
					#vgg-m
					"wc1": tf.get_variable("W0", shape=(7,7,params.IMAGE_CHANNELS,96), initializer=tf.contrib.layers.xavier_initializer()),
					"wc2": tf.get_variable("W1", shape=(5,5,96,256), initializer=tf.contrib.layers.xavier_initializer()),
					"wc3": tf.get_variable("W2", shape=(3,3,256,384), initializer=tf.contrib.layers.xavier_initializer()),
					"wc4": tf.get_variable("W3", shape=(3,3,384,384), initializer=tf.contrib.layers.xavier_initializer()),
					"wc5": tf.get_variable("W4", shape=(3,3,384,256), initializer=tf.contrib.layers.xavier_initializer()),
					"wd1": tf.get_variable("W5", shape=(6*6*256, 4096), initializer=tf.contrib.layers.xavier_initializer()),
					"wd2": tf.get_variable("W6", shape=(4096, 4096), initializer=tf.contrib.layers.xavier_initializer()),
					"wd3": tf.get_variable("W7", shape=(4096, params.N_CLASSES), initializer=tf.contrib.layers.xavier_initializer()),
					#grouping module
					"wd4": tf.get_variable("W8", shape=(6*6*256, 1), initializer=tf.contrib.layers.xavier_initializer())
				}
			with tf.name_scope("biases"):
				biases = {
					'bc1': tf.get_variable('B0', shape=(96), initializer=tf.contrib.layers.xavier_initializer()),
					'bc2': tf.get_variable('B1', shape=(256), initializer=tf.contrib.layers.xavier_initializer()),
					'bc3': tf.get_variable('B2', shape=(384), initializer=tf.contrib.layers.xavier_initializer()),
					'bc4': tf.get_variable('B3', shape=(384), initializer=tf.contrib.layers.xavier_initializer()),
					'bc5': tf.get_variable('B4', shape=(256), initializer=tf.contrib.layers.xavier_initializer()),
					'bd1': tf.get_variable('B5', shape=(4096), initializer=tf.contrib.layers.xavier_initializer()),
					'bd2': tf.get_variable('B6', shape=(4096), initializer=tf.contrib.layers.xavier_initializer()),
					'bd3': tf.get_variable('B7', shape=(params.N_CLASSES), initializer=tf.contrib.layers.xavier_initializer()),
					#grouping module
					'bd4': tf.get_variable('B8', shape=(1), initializer=tf.contrib.layers.xavier_initializer())
				}
		return weights, biases

	def get_view_descriptors_and_scores(self, x):
		"""Calculates view descriptors and view scores of each view. This is called for each batch element.

		Args:
			x: placeholder for input values; shape [n_views, image_size, image_size, 1]

		Returns:
			view_descriptors: view descriptors of batch element for all views; shape [n_views, 1, 6, 6, 512]
			view_discrimination_scores: view discriminations scores of element for all views; shape [n_views, 1, 1]

		"""
		view_descriptors = [] #n_views * [1, 6, 6, 512]
		view_discrimination_scores = [] #n_views * [1, 1]
		weights, biases = self.get_weights()
		#split across views to get list of tensors of each view
		views = tf.split(x, num_or_size_splits=params.N_VIEWS, axis=0) #n_views * [1, image_size, image_size, 1]
		for i in views:
			view_descriptor = self.cnn_convs(i, weights, biases)
			view_descriptors.append(view_descriptor)
			view_discrimination_score = self.cnn_grouping(view_descriptor, weights, biases)
			view_discrimination_scores.append(view_discrimination_score)
		return tf.convert_to_tensor(view_descriptors), tf.convert_to_tensor(view_discrimination_scores)

	def get_shape_descriptors(self, group_descriptors):
		"""Generates shape descriptors from a given batch element containing group descriptors. This is called
		for every batch element.

	  	Args:
	   		group_descriptors: pooled results of view descriptors divided in groups;
	   						   shape [n_views, 1, 6, 6, 512]

	   	Returns:
			shape_descriptors: shape descriptor after fc7 of each group, for simplicity it is
							   padded with zero descriptors to match amount of views;
							   shape [n_views, 1, 4096]
		"""
		shape_descriptors = []
		weights, biases = self.get_weights()
		with tf.name_scope("dropout"):
			dropout_prob = tf.placeholder_with_default(0.0, ())
		#get the shape descriptor i.e. the output of fc7 of every group descriptor
		for i in tf.split(group_descriptors, num_or_size_splits=params.N_VIEWS, axis=0):
			fc1 = tf.reshape(i, [-1, weights['wd1'].get_shape().as_list()[0]])
			fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
			fc1 = tf.nn.relu(fc1)
			fc1 = tf.nn.dropout(fc1, keep_prob=1-dropout_prob)

			fc2 = tf.reshape(fc1, [-1, weights['wd2'].get_shape().as_list()[0]])
			fc2 = tf.add(tf.matmul(fc2, weights['wd2']), biases['bd2'])
			fc2 = tf.nn.relu(fc2)
			fc2 = tf.nn.dropout(fc2, keep_prob=1-dropout_prob)

			shape_descriptors.append(fc2)

		return tf.convert_to_tensor(shape_descriptors)

	def get_learning_rate(self, find_lr=None):
		learning_rate = None
		if find_lr is None:
			learning_rate = params.LEARNING_RATE
		else:
			learning_rate = params.FIND_LEARNING_RATE_MIN

		return learning_rate

	def update_learning_rate(self, current_lr, find_lr=None):
		learning_rate = current_lr
		if find_lr is None:
			if params.LEARNING_RATE_TYPE == 0:
				pass
			elif params.LEARNING_RATE_TYPE == 1:
				pass
			elif params.LEARNING_RATE_TYPE == 2:
				pass
			elif params.LEARNING_RATE_TYPE == 3:
				pass
		else:
			learning_rate = current_lr * params.FIND_LEARNING_RATE_GROWTH

		return learning_rate

	# Wrapper functions for layer operations

	def conv2d(self, x, W, b, strides=1, padding="SAME", name=None):
		"""Applies convolution and bias addition and relu to inputs."""
		x = tf.nn.conv2d(x, W, strides=[1, strides, strides,1], padding=padding, name=name)
		x = tf.nn.bias_add(x, b)
		return tf.nn.relu(x)

	def max_pool2d(self, x, size=3, strides=2, padding="SAME"):
		return tf.nn.max_pool(x, ksize=[1, size, size, 1], strides=[1, strides, strides, 1], padding=padding)