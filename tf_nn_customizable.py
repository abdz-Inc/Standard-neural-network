from numpy.lib.function_base import gradient
import tensorflow as tf
import pickle
import numpy as np
import matplotlib.pyplot as plt


#initialize parameters for given layer dims
def initialize_params(layer_dims):

    params = {}
    L = len(layer_dims)
    init = tf.keras.initializers.GlorotNormal(seed = 1)

    for i in range(1, L):

        params['W'+str(i)] = tf.Variable(init(shape = (layer_dims[i], layer_dims[i-1])))
        params['b'+str(i)] = tf.Variable(init(shape = (layer_dims[i], 1)))

    return params


@tf.function
def forward_prop(X, params):

    A = X
    L = len(params)//2

    for i in range(L-1):
        Z = tf.math.add(tf.linalg.matmul(params['W'+str(i+1)], A), params['b'+str(i+1)])
        A = tf.keras.activations.relu(Z, threshold = 0.1)

    Z = tf.math.add(tf.linalg.matmul(params['W'+str(L)], A), params['b'+str(L)])
    A = tf.keras.activations.softmax(Z)

    return Z


@tf.function
def compute_cost(pred, true, cost_func):

    if cost_func == "binary_crossentrophy":
        cost = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_pred=pred, y_true=true, from_logits = True))
    
    elif cost_func == "categorical_crossentrophy":
        cost = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_pred=pred, y_true=true, from_logits = True))
    else:
        assert cost_func == "binary_crossentrophy" or cost_func == "categorical_crossentrophy", "choose an available cost funtion\noptions :\nbinary_crossentrophy\ncategorical_crossentrophy"
        
    return cost


def model(X, Y, layer_dims, optimizer = "adam", learning_rate = 0.0001, momentum = 0.9, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, minibatch_size = 64, num_epochs = 2500, cost_func = "binary_crossentrophy"):

    optimizer = optimizer.lower()
    optimizer_obj = None

    if optimizer == "adam":
        optimizer_obj = tf.keras.optimizers.Adam(learning_rate, beta_1=beta1, beta_2=beta2, epsilon=epsilon)

    elif optimizer == "rmsprop":
        optimizer_obj = tf.keras.optimizers.RMSprop(learning_rate, momentum = momentum, epsilon=epsilon)

    elif optimizer == "adamax":
        optimizer_obj = tf.keras.optimizers.Adamax(learning_rate, beta_1=beta1, beta_2=beta2, epsilon=epsilon)

    elif optimizer == "sgd":
        optimizer_obj = tf.keras.optimizers.SGD(learning_rate, momentum=momentum)

    
    assert optimizer_obj != None, "Enter a valid optimizer;\noptions :\nadam\nrmsprop\nadamax\nsgd" 
        
    costs = []
    params = initialize_params(layer_dims)

    X = X.batch(minibatch_size, drop_remainder = True).prefetch(8)
    Y = Y.batch(minibatch_size, drop_remainder = True).prefetch(8)

    for j in range(num_epochs):

        for mini_x, mini_y in zip(X, Y):
            
            trainables = []
            with tf.GradientTape() as tape:
                A = forward_prop(mini_x, params)
                cost = compute_cost(A, mini_y, cost_func)

            for i in range(len(params)//2):
                trainables.append(params['W'+str(i+1)])
                trainables.append(params['b'+str(i+1)])

            grads = tape.gradient(cost, trainables)
            optimizer_obj.apply_gradients(zip(grads, trainables))

        if j%10 == 0:
            costs.append(cost)
            print("Cost after epoch %i: %f" % (j, cost))

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per fives)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    with open('tf_params.pkl', 'wb') as f:
        pickle.dump(params, f)

    print('parameters saved')

    return params