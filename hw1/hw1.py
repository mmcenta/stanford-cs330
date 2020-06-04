from collections import defaultdict
import os
import pickle

import numpy as np
import random
import tensorflow as tf
from load_data import DataGenerator
from tensorflow.python.platform import flags
from tensorflow.keras.layers import LSTM, ConvLSTM2D, TimeDistributed, MaxPool2D, Flatten


FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'num_classes', 5, 'number of classes used in classification (e.g. 5-way classification).')

flags.DEFINE_integer('num_samples', 1,
                     'number of examples used for inner gradient update (K for K-shot learning).')

flags.DEFINE_integer('meta_batch_size', 16,
                     'Number of N-way classification tasks per batch.')

flags.DEFINE_string('logdir', './logs',
                    'Directory where logs will be saved.')

flags.DEFINE_string('name', '',
                    'Name of the run (used when saving logs).')

flags.DEFINE_string('label', '',
                    'Label of this run in plots.')

flags.DEFINE_integer('n_layers', 1,
                     'Number of hidden layers in the model.')

flags.DEFINE_integer('layer_units', 128,
                     'Number of units in the hidden layers.')

flags.DEFINE_bool('cnn', False,
                  'Whether to use the CNN-based architecture.')

flags.DEFINE_float('lr', 0.001,
                   'Learning rate.')

flags.DEFINE_float('dropout', 0.1,
                  'Whether to use dropout.')


def loss_function(preds, labels):
    """
    Computes MANN loss
    Args:
        preds: [B, K+1, N, N] network output
        labels: [B, K+1, N, N] labels
    Returns:
        scalar loss
    """
    #############################
    #### YOUR CODE GOES HERE ####
    last_N_preds = preds[:, -1:]
    last_N_labels = labels[:, -1:]
    loss = tf.losses.softmax_cross_entropy(last_N_labels, last_N_preds)
    return tf.reduce_mean(loss)
    #############################


class MANN(tf.keras.Model):
    def __init__(self, num_classes, samples_per_class,
                n_layers=1, layer_units=128):
        super(MANN, self).__init__()
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class

        self.hidden_layers = []
        for _ in range(n_layers):
            self.hidden_layers.append(LSTM(layer_units, return_sequences=True))
        self.output_layer = LSTM(num_classes, return_sequences=True)

    def call(self, input_images, input_labels):
        """
        MANN
        Args:
            input_images: [B, K+1, N, 784] flattened images
            labels: [B, K+1, N, N] ground truth labels
        Returns:
            [B, K+1, N, N] predictions
        """
        #############################
        #### YOUR CODE GOES HERE ####
        _, K, N, D = input_images.shape
        K -= 1

        # zero the last labels per batch (no slice assign on this tf version)
        input_labels = tf.concat(
            (input_labels[:, :-1], tf.zeros_like(input_labels[:, -1:])), axis=1)

        # concatenate the image vectors the label vectors
        x = tf.concat((input_images, input_labels), axis=-1)

        # unroll axes 1 and 2 into one 'timesteps' axis
        x = tf.reshape(x, (-1, (K + 1) * N, D + N))

        # feed the input to the network
        for layer in self.hidden_layers:
            x = layer(x)
        out = self.output_layer(x)

        # reshape to match the output shape
        out = tf.reshape(out, (-1, K + 1, N, N))
        #############################
        return out


class ConvMANN(tf.keras.Model):
    def __init__(self, num_classes, samples_per_class, dropout=0.1):
        super(ConvMANN, self).__init__()

        self.num_classes = num_classes
        self.samples_per_class = samples_per_class

        self.conv1 = ConvLSTM2D(16, 3, activation='relu', dropout=dropout,
            recurrent_dropout=2 * dropout, return_sequences=True)
        self.pool1 = TimeDistributed(MaxPool2D())

        self.conv2 = ConvLSTM2D(32, 3, activation='relu', dropout=dropout,
            recurrent_dropout=2 * dropout, return_sequences=True)
        self.pool2 = TimeDistributed(MaxPool2D())


        self.flatten = TimeDistributed(Flatten())

        self.lstm1 = LSTM(256, return_sequences=True)
        self.lstm2 = LSTM(num_classes, return_sequences=True)

    def call(self, input_images, input_labels):
        _, K, N, D, _ = input_images.shape
        K -= 1

        # zero the last labels per batch (no slice assign on this tf version)
        input_labels = tf.concat(
            (input_labels[:, :-1], tf.zeros_like(input_labels[:, -1:])), axis=1)

        # feed the input images to the CNN layers
        input_images = tf.reshape(input_images,
            (-1, (K + 1) * N, D, D)) # unroll
        input_images = tf.expand_dims(input_images,
            axis=-1)

        x_img = self.conv1(input_images)
        x_img = self.pool1(x_img)
        x_img = self.conv2(x_img)
        x_img = self.pool2(x_img)

        x_img = self.flatten(x_img)

        # append label vectors to feature vector
        input_labels = tf.reshape(input_labels,
            (-1, (K + 1) * N, N)) # unroll
        x = tf.concat((x_img, input_labels), axis=-1)

        # feed the feature vector to the LSTM
        x = self.lstm1(x)
        out = self.lstm2(x)

        # reshape to match the output shape
        out = tf.reshape(out, (-1, K + 1, N, N))
        return out


if FLAGS.cnn:
    ims = tf.placeholder(tf.float32, shape=(
        None, FLAGS.num_samples + 1, FLAGS.num_classes, 28, 28))
else:
    ims = tf.placeholder(tf.float32, shape=(
        None, FLAGS.num_samples + 1, FLAGS.num_classes, 784))
labels = tf.placeholder(tf.float32, shape=(
    None, FLAGS.num_samples + 1, FLAGS.num_classes, FLAGS.num_classes))

data_generator = DataGenerator(
    FLAGS.num_classes, FLAGS.num_samples + 1, config={"flatten": not FLAGS.cnn})

if FLAGS.cnn:
    o = ConvMANN(FLAGS.num_classes, FLAGS.num_samples + 1, FLAGS.dropout)
else:
    o = MANN(FLAGS.num_classes, FLAGS.num_samples + 1,
        n_layers=FLAGS.n_layers, layer_units=FLAGS.layer_units)
out = o(ims, labels)

loss = loss_function(out, labels)
optim = tf.train.AdamOptimizer(FLAGS.lr)
optimizer_step = optim.minimize(loss)

logs = defaultdict(list) # minimalist logging for plots
with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    for step in range(50000):
        i, l = data_generator.sample_batch('train', FLAGS.meta_batch_size)
        feed = {ims: i.astype(np.float32), labels: l.astype(np.float32)}
        _, ls = sess.run([optimizer_step, loss], feed)

        if step % 100 == 0:
            print("*" * 5 + "Iter " + str(step) + "*" * 5)
            i, l = data_generator.sample_batch('test', 100)
            feed = {ims: i.astype(np.float32),
                    labels: l.astype(np.float32)}
            pred, tls = sess.run([out, loss], feed)
            print("Train Loss:", ls, "Test Loss:", tls)
            pred = pred.reshape(
                -1, FLAGS.num_samples + 1,
                FLAGS.num_classes, FLAGS.num_classes)
            pred = pred[:, -1, :, :].argmax(2)
            l = l[:, -1, :, :].argmax(2)
            test_accuracy = (1.0 * (pred == l)).mean()
            print("Test Accuracy", test_accuracy)

            # log metrics
            logs['iteration'].append(step)
            logs['train_loss'].append(ls)
            logs['test_loss'].append(tls)
            logs['test_accuracy'].append(test_accuracy)

# dump logs
logs['name'] = FLAGS.name
logs['label'] = FLAGS.label
os.makedirs(FLAGS.logdir, exist_ok=True)
logfile = '{}_N={}_K={}_B={}.pkl'.format(
    FLAGS.name, FLAGS.num_classes, FLAGS.num_samples, FLAGS.meta_batch_size)
with open(os.path.join(FLAGS.logdir, logfile), "wb") as f:
    pickle.dump(logs, f)
