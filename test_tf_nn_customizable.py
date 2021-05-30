import tensorflow as tf
import numpy as np
import pickle
from tf_nn_customizable import model
import h5py

'''
def load_data():
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
'''


def get_dataset():
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    test_dataset = h5py.File('datasets/test_signs.h5', "r")

    x_train = tf.data.Dataset.from_tensor_slices(train_dataset['train_set_x'])
    y_train = tf.data.Dataset.from_tensor_slices(train_dataset['train_set_y'])

    x_test = tf.data.Dataset.from_tensor_slices(test_dataset['test_set_x'])
    y_test = tf.data.Dataset.from_tensor_slices(test_dataset['test_set_y'])

    def normalize(image):
        
        image = tf.cast(image, tf.float32) / 256.0
        image = tf.reshape(image, [-1,1])
        return image

    def one_hot_matrix(label, depth=6):
        
        one_hot = tf.reshape(tf.one_hot(label, depth, axis = 0),(depth, 1))
        
        return one_hot

    new_y_test = y_test.map(one_hot_matrix)
    new_y_train = y_train.map(one_hot_matrix)

    new_train = x_train.map(normalize)
    new_test = x_test.map(normalize)

    return new_train, new_y_train, new_test, new_y_test

new_train, new_y_train, new_test, new_y_test = get_dataset()

for i in new_train:
    dim1 = i.numpy().shape[0]
    break
print(dim1)
params = model(new_train, new_y_train,(dim1, 20, 16, 8, 6),learning_rate= 0.01,optimizer = "adam", minibatch_size = 892, num_epochs = 500)