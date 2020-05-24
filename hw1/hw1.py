from collections import defaultdict
import os
import pickle

import numpy as np
import random
import tensorflow as tf
from load_data import DataGenerator
from tensorflow.python.platform import flags
from tensorflow.keras import layers


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

flags.DEFINE_integer('layer_size', 128,
                     'Number of units in the hidden layers.')


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
            self.hidden_layers.append(tf.keras.layers.LSTM(layer_units,
                return_sequences=True))
        self.output_layer = tf.keras.layers.LSTM(num_classes, return_sequences=True)

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
        B, K, N, D = input_images.shape
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

ims = tf.placeholder(tf.float32, shape=(
    None, FLAGS.num_samples + 1, FLAGS.num_classes, 784))
labels = tf.placeholder(tf.float32, shape=(
    None, FLAGS.num_samples + 1, FLAGS.num_classes, FLAGS.num_classes))

data_generator = DataGenerator(
    FLAGS.num_classes, FLAGS.num_samples + 1)

o = MANN(FLAGS.num_classes, FLAGS.num_samples + 1)
out = o(ims, labels)

loss = loss_function(out, labels)
optim = tf.train.AdamOptimizer(0.001)
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
