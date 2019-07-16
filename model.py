import os
import numpy as np
import tensorflow as tf
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
			activations = conv1
			conv1 = self.max_pool2d(conv1, size=3, strides=2, padding="VALID") #(1, 54, 54, 96)
		with tf.name_scope("conv2"):
			conv2 = self.conv2d(conv1, weights['wc2'], biases['bc2'], strides=2, padding="SAME") #(1, 27, 27, 256)
			#activations = conv2
			conv2 = self.max_pool2d(conv2, size=3, strides=2, padding="VALID") #(1, 13, 13, 256)
		with tf.name_scope("conv3"):
			conv3 = self.conv2d(conv2, weights['wc3'], biases['bc3'], strides=1, padding="SAME") #(1, 13, 13, 512)
			#activations = conv3
			#conv3 = max_pool2d(conv3, size=3) #
		with tf.name_scope("conv4"):
			conv4 = self.conv2d(conv3, weights['wc4'], biases['bc4'], strides=1, padding="SAME") #(1, 13, 13, 512)
			#activations = conv4
			#conv4 = max_pool2d(conv4, size=3) #
		with tf.name_scope("conv5"):
			conv5 = self.conv2d(conv4, weights['wc5'], biases['bc5'], strides=1, padding="SAME") #(1, 13, 13, 512)
			#activations = conv5
			conv5 = self.max_pool2d(conv5, size=3, strides=2, padding="VALID") #(1, 6, 6, 512)
		return conv5, activations

	def cnn_fcs(self, x, weights, biases, dropout_prob):
		"""Applies three fully connected layer operations to input
		according to vgg-m architecture.

		Args:
			x: input (output of conv5)
			weights: weights of filter
			biases: biases of layers
			dropout_prob: identical dropout probabilities; shape [-1, 1]

		Returns:
			fc3: input membership to classes, i.e. predicted classes
		"""
		with tf.name_scope("fc1"):
			fc1 = tf.reshape(x, [-1, weights['wd1'].get_shape().as_list()[0]])
			fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
			fc1 = tf.nn.relu(fc1) #(1, 4096)
			fc1 = tf.nn.dropout(fc1, keep_prob=1-dropout_prob[0][0])
		with tf.name_scope("fc2"):
			fc2 = tf.reshape(fc1, [-1, weights['wd2'].get_shape().as_list()[0]])
			fc2 = tf.add(tf.matmul(fc2, weights['wd2']), biases['bd2'])
			fc2 = tf.nn.relu(fc2) #(1, 4096)
			fc2 = tf.nn.dropout(fc2, keep_prob=1-dropout_prob[0][0])
		with tf.name_scope("fc3"):
			#fc3 = tf.reshape(fc2, [-1, weights['wd3'].get_shape().as_list()[0]])
			fc3 = tf.add(tf.matmul(fc2, weights['wd3']), biases['bd3']) #(1,n_classes)

		return fc3

	def cnn_fcs_grouping(self, group_descriptors, group_weights, weights, biases, dropout_prob):
		"""Classifies each batch element object from views by taking group descriptors of whole batch as input.

		Args:
			group_descriptors: group descriptors of pooled views, for simplicity
							   they are padded with zero descriptors to match amount
							   of views; shape [-1, n_views, 1, 6, 6, 512]
			group_weights: weight of each group; shape [-1, views]
			weights: model weights
			biases: model biases
			dropout_prob: identical dropout probilities; shape [-1, 1]

		Returns:
			fc3: predicted classification of object or views, respectively; shape [-1, 1, n_classes]
		"""
		shape_descriptors = tf.map_fn(self.get_shape_descriptors, [group_descriptors, dropout_prob], dtype=tf.float32)
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

	def cnn_grouping(self, x, weights, biases, dropout_prob):
		"""Calculates the view discrimination score of a view.

		Args:
			x: view descriptor as input; shape [1, 6, 6, 512]
			weights: weights of cnn
			biases: biases of cnn
			dropout_prob: list of dropout probability;

		Returns:
			fc1: view discrimination score of input view
		"""
		with tf.name_scope("fc4"):
			fc1 = tf.reshape(x, [-1, weights['wd4'].get_shape().as_list()[0]])
			fc1 = tf.add(tf.matmul(fc1, weights['wd4']), biases['bd4'])
			fc1 = tf.nn.leaky_relu(fc1)
			#fc1 = tf.nn.dropout(fc1, 1-dropout_prob[0])
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
		global_step = tf.Variable(0, trainable=False, name="global_step")
		x_train, y_train, x_test, y_test = dataset
		is_multi_view = False
		# if shape of x placeholder relates to n_views dimension a multi-view input is supposed
		if x.shape[1] == params.N_VIEWS:
			is_multi_view = True
			batch_size = params.BATCH_SIZE_MULTI
			dropout_prob = tf.placeholder(tf.float32, shape=(None, 1), name="dropout_prob")
			# feed each batch element of input in first part of cnn (conv layers) to get view descriptors and scores of each channel, i.e. view
			# view_descriptors = [-1, n_views, 1, 6, 6, 512], view_discrimination_scores = [-1, n_views, 1, 1] -> [-1, n_views]
			with tf.name_scope("view_descriptors_and_scores"):
				view_descriptors, view_discrimination_scores, activations_convs = tf.map_fn(self.get_view_descriptors_and_scores, [x, dropout_prob], dtype=(tf.float32, tf.float32, tf.float32))
				#make scores more compact
				view_discrimination_scores = tf.reshape(view_discrimination_scores, [-1, params.N_VIEWS])

			with tf.name_scope("group_weights"):
				batch_group_idx, batch_group_weights = tf.map_fn(self.grouping.get_group_weights, view_discrimination_scores, dtype=(tf.int32, tf.float32))

			with tf.name_scope("grouping"):
				batch_group_descriptors = tf.map_fn(self.grouping.get_group_descriptors, [batch_group_idx, view_descriptors], dtype=tf.float32)

			with tf.name_scope("fully-connected"):
				pred = self.cnn_fcs_grouping(batch_group_descriptors, batch_group_weights, weights, biases, dropout_prob)
		else:
			batch_size = params.BATCH_SIZE_SINGLE
			dropout_prob = tf.placeholder(tf.float32, shape=(None, 1), name="dropout_prob")
			# feed input to connected cnn
			pred = self.cnn_convs(x, weights, biases)
			pred = self.cnn_fcs(pred, weights, biases, dropout_prob)

		#sigmoid for multi-label, softmax for single-label
		if params.DATASET_IS_SINGLELABEL:
			cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
		else:
			cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y))

		if find_lr:
			learning_rate = tf.placeholder(tf.float32, ())
			optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
		else:
			learning_rate = params.LEARNING_RATE
			optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)


		if params.DATASET_IS_SINGLELABEL:
			correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
		else:
			correct_prediction = tf.equal(tf.round(tf.sigmoid(pred)), y)		
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		init = tf.global_variables_initializer()
		saver = tf.train.Saver()

		with tf.name_scope("summaries"):
			summary_loss = tf.summary.scalar("loss", cost)
			summary_accuracy = tf.summary.scalar("accuracy", accuracy)
			merged = tf.summary.merge_all()

		#config = tf.ConfigProto()
		#config.gpu_options.allow_growth = True
		with tf.Session() as sess:
			if find_lr:
				lr = self.get_learning_rate(find_lr)
			if ckpt is None:
				sess.run(init)
			else:
				saver.restore(sess, ckpt)

			train_summary_writer = tf.summary.FileWriter(os.path.join(params.SUMMARY_PATH, "train"), sess.graph)
			test_summary_writer = tf.summary.FileWriter(os.path.join(params.SUMMARY_PATH, "test"), sess.graph)

			train_batch_loss = []
			train_batch_accuracy = []
			epochs_train_accuracy = []
			epochs_train_loss = []
			epochs_test_loss = []
			epochs_test_accuracy = []
			learning_rates = []

			for i in range(params.TRAINING_EPOCHS if find_lr is None else 100):
				x_train, y_train = self.data.shuffle(x_train, y_train)
				x_test, y_test = self.data.shuffle(x_test, y_test)

				for batch in range(int(np.ceil(len(x_train)/batch_size))):
					print("Epoch", i, "Batch", batch, "of", len(x_train)//batch_size, end="\r")
					batch_x = x_train[batch*batch_size:min((batch+1)*batch_size,len(x_train))]
					batch_y = y_train[batch*batch_size:min((batch+1)*batch_size,len(y_train))]

					if params.DATASET_LOAD_DYNAMIC:
						batch_x, batch_y = self.data.load_dynamic_dataset(batch_x, batch_y, batch)

					if find_lr:
						learning_rates.append(lr)
					# Run optimization op (backprop).
					    # Calculate batch loss and accuracy
					dropout_prob_value = np.full([len(batch_x), 1], params.DROPOUT_PROB)
					opt = None
					acc = None
					loss = None
					if is_multi_view:
						if params.USE_SUMMARY:
							if find_lr:
								summary, opt = sess.run([merged, optimizer], feed_dict={x: batch_x, y: batch_y, dropout_prob: dropout_prob_value, learning_rate: lr})
							else:
								summary, opt = sess.run([merged, optimizer], feed_dict={x: batch_x, y: batch_y, dropout_prob: dropout_prob_value})
								#train_summary_writer.add_summary(summary, tf.train.global_step(sess, global_step))
						else:
							if find_lr:
								opt = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, dropout_prob: dropout_prob_value, learning_rate: lr})
							else:
								opt = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, dropout_prob: dropout_prob_value})
						if find_lr:
							loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, dropout_prob: np.zeros([len(batch_x), 1]), learning_rate: lr})
						else:
							loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, dropout_prob: np.zeros([len(batch_x), 1])})							

					else:
						if find_lr:
							opt = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, learning_rate: lr})
							loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, learning_rate: lr})
						else:
							opt = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, dropout_prob: dropout_prob_value})
							loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, dropout_prob: np.zeros([len(batch_x), 1])})

					train_batch_loss.append(loss)
					train_batch_accuracy.append(acc)

					#finding learning rate is active, therefore break if latest loss is much larger than loss before
					if find_lr is not None and i > 0:
						if train_batch_loss[-1] > 4 * train_batch_loss[-3]:
							return train_batch_loss, train_batch_accuracy, epochs_test_accuracy, learning_rates
					
					if find_lr:
						lr = self.update_learning_rate(lr, find_lr)

				#Calculate accuracy over whole training set
				train_batch_sizes = []
				epoch_train_acc = []
				epoch_train_loss = []
				for batch in range(int(np.ceil(len(x_train)/batch_size))):
					batch_x = x_train[batch*batch_size:min((batch+1)*batch_size,len(x_train))]
					batch_y = y_train[batch*batch_size:min((batch+1)*batch_size,len(y_train))]
					if params.DATASET_LOAD_DYNAMIC:
						batch_x, batch_y = self.data.load_dynamic_dataset(batch_x, batch_y, batch)
					train_batch_sizes.append(len(x_train))
					if is_multi_view:
						if params.USE_SUMMARY:
							train_acc, train_loss, summary = sess.run([accuracy, cost, merged], feed_dict={x: batch_x, y : batch_y, dropout_prob: np.zeros([len(batch_x), 1])})
							train_summary_writer.add_summary(summary)
						else:
							train_acc, train_loss = sess.run([accuracy, cost], feed_dict={x: batch_x, y : batch_y, dropout_prob: np.zeros([len(batch_x), 1])})

					epoch_train_acc.append(train_acc)
					epoch_train_loss.append(train_loss)

				epochs_train_accuracy.append(np.average(epoch_train_acc, weights=train_batch_sizes))
				epochs_train_loss.append(np.average(epoch_train_loss, weights=train_batch_sizes))

				print("Epoch {}, Loss {:.6f}, Last Batch Loss {:.6f}, Training Accuracy {:.5f}".format(i, epochs_train_loss[-1], train_batch_loss[-1], epochs_train_accuracy[-1]))
				
				#Calculate accuracy for all validation/testing images
				#divide in batches and calculate mean to prevent OOM error
				if find_lr is None:
					batch_test_sizes = []
					test_batches_accuracy = []
					test_batches_loss = []
					for test_batch in range(int(np.ceil(len(x_test)/batch_size))):
						batch_x_test = x_test[test_batch*batch_size:min((test_batch+1)*batch_size, len(x_test))]
						batch_y_test = y_test[test_batch*batch_size:min((test_batch+1)*batch_size, len(y_test))]
						if params.DATASET_LOAD_DYNAMIC:
							batch_x_test, batch_y_test = self.data.load_dynamic_dataset(batch_x_test, batch_y_test, test_batch)
						batch_test_sizes.append(len(batch_x_test))
						if is_multi_view:
							if params.USE_SUMMARY:
								test_acc, test_loss, summary = sess.run([accuracy, cost, merged], feed_dict={x: batch_x_test, y : batch_y_test, dropout_prob: np.zeros([len(batch_x_test), 1])})
							else:
								test_acc, test_loss = sess.run([accuracy, cost], feed_dict={x: batch_x_test, y : batch_y_test, dropout_prob: np.zeros([len(batch_x_test), 1])})
						else:
							test_acc, test_loss = sess.run([accuracy,cost], feed_dict={x: batch_x_test, y : batch_y_test, dropout_prob: np.zeros([len(batch_x_test), 1])})

						test_batches_loss.append(test_loss)
						test_batches_accuracy.append(test_acc)

					#only save latest summary
					if params.USE_SUMMARY:
						test_summary_writer.add_summary(summary, tf.train.global_step(sess, global_step))

					#weight the loss and accuracy with the current batch size
					epoch_test_accuracies_avg = np.average(test_batches_accuracy, weights=batch_test_sizes)
					epoch_test_losses_avg = np.average(test_batches_loss, weights=batch_test_sizes)
					epochs_test_accuracy.append(epoch_test_accuracies_avg)
					epochs_test_loss.append(epoch_test_losses_avg)
					print("Testing Loss: {:.6f}, Testing Accuracy: {:.5f}".format(epoch_test_losses_avg, epoch_test_accuracies_avg))
			try:
				saver.save(sess, os.path.join(params.CKPT_PATH, params.CKPT_FILE))
			except Exception as e:
				print(e)
			print("Training finished!")

			train_summary_writer.close()
			test_summary_writer.close()

		return train_batch_loss, train_batch_accuracy, epochs_train_loss, epochs_train_accuracy, epochs_test_accuracy, epochs_test_loss, learning_rate

	def predict(self, img, labels=None, get_saliency=True, get_activations=True, n_views=params.N_VIEWS, ckpt_file=None):
		weights, biases = self.get_weights()
		is_multi_view = None
		saliencies = []
		groups = []
		view_scores = []
		group_ids = []
		group_weights = []
		correct_predictions = []
		is_corrects = []
		pred_label_ids = []
		classifications = []
		acts_convs = []
		
		#if multi view object is given
		if img.shape[1] == n_views:
			is_multi_view = True
			batch_size = params.BATCH_SIZE_MULTI
			x = tf.placeholder(tf.float32, [None, n_views, params.IMAGE_SIZE, params.IMAGE_SIZE, params.IMAGE_CHANNELS], name="x")
			dropout_prob = tf.placeholder(tf.float32, shape=(None, 1), name="dropout_prob")

			with tf.name_scope("view_descriptors_and_scores"):
				view_descriptors, view_discrimination_scores, activations_convs = tf.map_fn(self.get_view_descriptors_and_scores, [x, dropout_prob], dtype=(tf.float32, tf.float32, tf.float32))
				view_discrimination_scores = tf.reshape(view_discrimination_scores, [-1, n_views])

			with tf.name_scope("group_weights"):
				batch_group_idx, batch_group_weights = tf.map_fn(self.grouping.get_group_weights, view_discrimination_scores, dtype=(tf.int32, tf.float32))

			with tf.name_scope("grouping"):
				batch_group_descriptors = tf.map_fn(self.grouping.get_group_descriptors, [batch_group_idx, view_descriptors], dtype=tf.float32)
				pred = self.cnn_fcs_grouping(batch_group_descriptors, batch_group_weights, weights, biases, dropout_prob)

			with tf.name_scope("fully-connected"):
				saliency = tf.gradients(pred, x)
				if params.DATASET_IS_SINGLELABEL:
					pred = tf.nn.softmax(pred)
				else:
					pred = tf.nn.sigmoid(pred)

		else:
			is_multi_view = False
			batch_size = params.BATCH_SIZE_SINGLE
			dropout_prob = tf.placeholder(tf.float32, shape=(None, 1), name="dropout_prob")
			# feed input to connected cnn
			x = tf.placeholder(tf.float32, [None, params.IMAGE_SIZE, params.IMAGE_SIZE, params.IMAGE_CHANNELS], name="x")
			pred, activations_convs = self.cnn_convs(x, weights, biases)
			pred = self.cnn_fcs(pred, weights, biases, dropout_prob)
			saliency = tf.gradients(pred, x)
			pred = tf.nn.softmax(pred)

		y = tf.placeholder(tf.float32, [None, params.N_CLASSES])
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
			for batch in range(int(np.ceil(img.shape[0]/batch_size))):
				batch_x = img[batch*batch_size:min((batch+1)*batch_size,img.shape[0])]
				classification = sess.run(pred, feed_dict={x: batch_x, dropout_prob: np.zeros([batch_x.shape[0], 1])})
				classifications.extend(classification)
				if labels is not None:
					batch_y = labels[batch*batch_size:min((batch+1)*batch_size,labels.shape[0])]
					is_correct = sess.run(correct_prediction, feed_dict={x: batch_x, y: batch_y, dropout_prob: np.zeros([batch_x.shape[0], 1])})
					pred_label_id = np.argmax(classification, axis=1)
				else:
					batch_y = [None]*batch_x.shape[0]
					is_correct = [None]*batch_x.shape[0]
					pred_label_id = [None]*batch_x.shape[0]

				is_corrects.extend(is_correct)
				pred_label_ids.extend(pred_label_id)
				#correct_predictions.append((is_correct, np.argmax(batch_y, axis=1)))

				if is_multi_view:
					g_ids, g_weights, v_scores, act_convs, sal = sess.run([batch_group_idx, batch_group_weights, view_discrimination_scores, activations_convs, saliency], feed_dict={x: batch_x, dropout_prob: np.zeros([batch_x.shape[0], 1])})
					group_ids.extend(g_ids)
					group_weights.extend(g_weights)
					view_scores.extend(v_scores)
					if get_activations:
						acts_convs.append(act_convs)
					else:
						#save one activations object for later shape lookup
						acts_convs = [act_convs]
					if get_saliency:
						saliencies.append(sal)
			
			saliencies = np.reshape(saliencies, [-1, n_views, params.IMAGE_SIZE, params.IMAGE_SIZE, params.IMAGE_CHANNELS])
			groups =  np.reshape(groups, [-1, n_views])
			view_scores =  np.reshape(view_scores, [-1, n_views])
			group_ids =  np.reshape(group_ids, [-1, n_views])
			group_weights =  np.reshape(group_weights, [-1, n_views])
			correct_predictions =  np.reshape(list(zip(is_corrects, pred_label_ids)), [-1, 2])
			classifications =  np.reshape(classifications, [-1, params.N_CLASSES])
			acts_convs = np.reshape(acts_convs,  [-1, *acts_convs[0].shape[1:]])

		return classifications, saliencies, view_scores, group_ids, group_weights, correct_predictions, acts_convs

	def get_weights(self):
		with tf.variable_scope("cnn", reuse=tf.AUTO_REUSE):
			with tf.name_scope("weights"):
				weights = {
					#shape = (filter_size_row, filter_size_col, channels of input, number of convs)
					#vgg-m
					"wc1": tf.get_variable("W0", shape=(7,7,params.IMAGE_CHANNELS,96), initializer=tf.contrib.layers.variance_scaling_initializer()),
					"wc2": tf.get_variable("W1", shape=(5,5,96,256), initializer=tf.contrib.layers.variance_scaling_initializer()),
					"wc3": tf.get_variable("W2", shape=(3,3,256,384), initializer=tf.contrib.layers.variance_scaling_initializer()),
					"wc4": tf.get_variable("W3", shape=(3,3,384,384), initializer=tf.contrib.layers.variance_scaling_initializer()),
					"wc5": tf.get_variable("W4", shape=(3,3,384,256), initializer=tf.contrib.layers.variance_scaling_initializer()),
					"wd1": tf.get_variable("W5", shape=(6*6*256, 4096), initializer=tf.contrib.layers.variance_scaling_initializer()),
					"wd2": tf.get_variable("W6", shape=(4096, 4096), initializer=tf.contrib.layers.variance_scaling_initializer()),
					"wd3": tf.get_variable("W7", shape=(4096, params.N_CLASSES), initializer=tf.contrib.layers.variance_scaling_initializer()),
					#grouping module
					"wd4": tf.get_variable("W8", shape=(6*6*256, 1), initializer=tf.contrib.layers.variance_scaling_initializer())
				}
			with tf.name_scope("biases"):
				biases = {
					'bc1': tf.get_variable('B0', shape=(96), initializer=tf.contrib.layers.variance_scaling_initializer()),
					'bc2': tf.get_variable('B1', shape=(256), initializer=tf.contrib.layers.variance_scaling_initializer()),
					'bc3': tf.get_variable('B2', shape=(384), initializer=tf.contrib.layers.variance_scaling_initializer()),
					'bc4': tf.get_variable('B3', shape=(384), initializer=tf.contrib.layers.variance_scaling_initializer()),
					'bc5': tf.get_variable('B4', shape=(256), initializer=tf.contrib.layers.variance_scaling_initializer()),
					'bd1': tf.get_variable('B5', shape=(4096), initializer=tf.contrib.layers.variance_scaling_initializer()),
					'bd2': tf.get_variable('B6', shape=(4096), initializer=tf.contrib.layers.variance_scaling_initializer()),
					'bd3': tf.get_variable('B7', shape=(params.N_CLASSES), initializer=tf.contrib.layers.variance_scaling_initializer()),
					#grouping module
					'bd4': tf.get_variable('B8', shape=(1), initializer=tf.contrib.layers.variance_scaling_initializer())
				}
		return weights, biases

	def get_view_descriptors_and_scores(self, x_and_dropout_prob):
		"""Calculates view descriptors and view scores of each view. This is called for each batch element.

		Args:
			x_and_dropout_prob: tuple containing placeholder for input values; shape [n_views, image_size, image_size, 1]
				and dropout probability; shape [1]

		Returns:
			view_descriptors: view descriptors of batch element for all views; shape [n_views, 1, 6, 6, 512]
			view_discrimination_scores: view discriminations scores of element for all views; shape [n_views, 1, 1]

		"""
		view_descriptors = [] #n_views * [1, 6, 6, 512]
		view_discrimination_scores = [] #n_views * [1, 1]
		activations_convs = [] #n_views * [conv_shape]
		x, dropout_prob = x_and_dropout_prob
		weights, biases = self.get_weights()
		#split across views to get list of tensors of each view
		views = tf.split(x, num_or_size_splits=params.N_VIEWS, axis=0) #n_views * [1, image_size, image_size, 1]
		for i in views:
			view_descriptor, activations = self.cnn_convs(i, weights, biases)
			activations_convs.append(activations)
			view_descriptors.append(view_descriptor)
			view_discrimination_score = self.cnn_grouping(view_descriptor, weights, biases, dropout_prob)
			view_discrimination_scores.append(view_discrimination_score)
		return tf.convert_to_tensor(view_descriptors), tf.convert_to_tensor(view_discrimination_scores), tf.convert_to_tensor(activations_convs)

	def get_shape_descriptors(self, group_descriptors_and_dropout_prob):
		"""Generates shape descriptors from a given batch element containing group descriptors. This is called
		for every batch element.

	  	Args:
	   		group_descriptors_and_dropout_prob: tuple containing pooled results of view descriptors divided in groups;
	   						   shape [n_views, 1, 6, 6, 512] and list of dropout probability; shape [1]

	   	Returns:
			shape_descriptors: shape descriptor after fc7 of each group, for simplicity it is
							   padded with zero descriptors to match amount of views;
							   shape [n_views, 1, 4096]
		"""
		shape_descriptors = []
		weights, biases = self.get_weights()
		group_descriptors, dropout_prob = group_descriptors_and_dropout_prob
		#get the shape descriptor i.e. the output of fc7 of every group descriptor
		for i in tf.split(group_descriptors, num_or_size_splits=params.N_VIEWS, axis=0):
			fc1 = tf.reshape(i, [-1, weights['wd1'].get_shape().as_list()[0]])
			fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
			fc1 = tf.nn.relu(fc1)
			fc1 = tf.nn.dropout(fc1, keep_prob=1-dropout_prob[0])

			fc2 = tf.reshape(fc1, [-1, weights['wd2'].get_shape().as_list()[0]])
			fc2 = tf.add(tf.matmul(fc2, weights['wd2']), biases['bd2'])
			fc2 = tf.nn.relu(fc2)
			fc2 = tf.nn.dropout(fc2, keep_prob=1-dropout_prob[0])

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