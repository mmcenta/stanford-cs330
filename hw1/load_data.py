import numpy as np
import os
import random
import tensorflow as tf
from scipy import misc


def get_images(paths, labels, nb_samples=None, shuffle=True):
    """
    Takes a set of character folders and labels and returns paths to image files
    paired with labels.
    Args:
        paths: A list of character folders
        labels: List or numpy array of same length as paths
        nb_samples: Number of images to retrieve per character
    Returns:
        List of (label, image_path) tuples
    """
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images_labels = [(i, os.path.join(path, image))
                     for i, path in zip(labels, paths)
                     for image in sampler(os.listdir(path))]
    if shuffle:
        random.shuffle(images_labels)
    return images_labels


def image_file_to_array(filename, img_shape, flatten=True):
    """
    Takes an image path and returns numpy array
    Args:
        filename: Image filename
        img_shape: Shape of the image
    Returns:
        1 channel image
    """
    image = misc.imread(filename)
    if flatten:
        w, h = img_shape
        image = image.reshape([w * h])
    image = image.astype(np.float32) / 255.0
    image = 1.0 - image
    return image


class DataGenerator(object):
    """
    Data Generator capable of generating batches of Omniglot data.
    A "class" is considered a class of omniglot digits.
    """

    def __init__(self, num_classes, num_samples_per_class, config={}):
        """
        Args:
            num_classes: Number of classes for classification (K-way)
            num_samples_per_class: num samples to generate per class in one batch
            batch_size: size of meta batch size (e.g. number of functions)
        """
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = num_classes

        data_folder = config.get('data_folder', '../data/omniglot_resized')
        self.img_size = config.get('img_size', (28, 28))
        self.flatten = config.get('flatten', False)

        self.dim_input = np.prod(self.img_size)
        self.dim_output = self.num_classes

        character_folders = [os.path.join(data_folder, family, character)
                             for family in os.listdir(data_folder)
                             if os.path.isdir(os.path.join(data_folder, family))
                             for character in os.listdir(os.path.join(data_folder, family))
                             if os.path.isdir(os.path.join(data_folder, family, character))]

        random.seed(1)
        random.shuffle(character_folders)
        num_val = 100
        num_train = 1100
        self.metatrain_character_folders = character_folders[: num_train]
        self.metaval_character_folders = character_folders[
            num_train:num_train + num_val]
        self.metatest_character_folders = character_folders[
            num_train + num_val:]

    def sample_batch(self, batch_type, batch_size):
        """
        Samples a batch for training, validation, or testing
        Args:
            batch_type: train/val/test
        Returns:
            A a tuple of (1) Image batch and (2) Label batch where
            image batch has shape [B, K, N, 784] and label batch has shape [B, K, N, N]
            where B is batch size, K is number of samples per class, N is number of classes
        """
        if batch_type == "train":
            folders = self.metatrain_character_folders
        elif batch_type == "val":
            folders = self.metaval_character_folders
        else:
            folders = self.metatest_character_folders

        #############################
        #### YOUR CODE GOES HERE ####
        all_image_batches = []
        all_label_batches = []
        for _ in range(batch_size):
            # create batches of K lists
            images = [list() for _ in range(self.num_samples_per_class)]
            labels = [list() for _ in range(self.num_samples_per_class)]
            next_idx = [0] * self.num_classes

            # sample the classes and images
            classes = np.random.choice(folders, size=(self.num_classes,))
            labels_and_paths = get_images(classes, range(self.num_classes),
                nb_samples=self.num_samples_per_class)

            # load images and one-hot encode labels
            for label, path in labels_and_paths:
                # only add one class instance per sample list
                idx = next_idx[label]


                image = image_file_to_array(path, self.img_size, flatten=self.flatten)
                one_hot_label = np.zeros((self.num_classes,))
                one_hot_label[label] = 1.

                images[idx].append(image)
                labels[idx].append(one_hot_label)

                next_idx[label] += 1

            all_image_batches.append(images)
            all_label_batches.append(labels)

        # convert to numpy arrays
        all_image_batches = np.array(all_image_batches)
        all_label_batches = np.array(all_label_batches)
        #############################

        return all_image_batches, all_label_batches
