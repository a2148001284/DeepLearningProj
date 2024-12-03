import tensorflow as tf
import h5py
import matplotlib.pyplot as plt
import numpy as np
import math


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    m = X.shape[0]
    mini_batches = []
    np.random.seed(seed)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation]
    shuffled_Y = Y[permutation]

    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = X[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch_Y = Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:
        mini_batch_X = X[num_complete_minibatches * mini_batch_size: m]
        mini_batch_Y = Y[num_complete_minibatches * mini_batch_size: m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


def load_dataset():
    train_dataset = h5py.File('./data/signs/train_signs.h5', "r")
    X_train_orig = np.array(train_dataset["train_set_x"][:])
    Y_train_orig = np.array(train_dataset["train_set_y"][:])
    test_dataset = h5py.File('./data/signs/test_signs.h5', "r")
    X_test_orig = np.array(test_dataset["test_set_x"][:])
    Y_test_orig = np.array(test_dataset["test_set_y"][:])

    Y_train_orig = Y_train_orig.reshape((1, Y_train_orig.shape[0]))
    Y_test_orig = Y_test_orig.reshape((1, Y_test_orig.shape[0]))

    X_train = X_train_orig / 255.
    X_test = X_test_orig / 255.
    Y_train = convert_to_one_hot(Y_train_orig, 6).T
    Y_test = convert_to_one_hot(Y_test_orig, 6).T

    return X_train, Y_train, X_test, Y_test


def initialize_parameters():
    tf.random.set_seed(1)
    initializer = tf.keras.initializers.GlorotUniform(seed=0)
    W1 = tf.Variable(initializer([3, 3, 3, 8]), name="W1")
    W2 = tf.Variable(initializer([3, 3, 8, 16]), name="W2")

    parameters = {"W1": W1,
                  "W2": W2}

    return parameters


def forward_propagation(X, parameters):
    W1 = parameters['W1']
    W2 = parameters['W2']

    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool2d(A1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='SAME')
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool2d(A2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    F = tf.reshape(P2, shape=[-1, 16 * 16 * 16])
    Z3 = tf.keras.layers.Dense(6)(F)

    return Z3


def compute_cost(Z3, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))
    return cost


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.009, num_epochs=100, minibatch_size=64):
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]

    parameters = initialize_parameters()
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    costs = []
    seed = 3

    for epoch in range(num_epochs):
        minibatch_cost = 0.
        seed = seed + 1
        minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

        for minibatch_X, minibatch_Y in minibatches:
            with tf.GradientTape() as tape:
                Z3 = forward_propagation(minibatch_X, parameters)
                cost = compute_cost(Z3, minibatch_Y)

            grads = tape.gradient(cost, list(parameters.values()))
            optimizer.apply_gradients(zip(grads, list(parameters.values())))
            minibatch_cost += cost / len(minibatches)

        if epoch % 5 == 0:
            print(f"迭代次数： {epoch}: 损失大小为：{minibatch_cost}")
            costs.append(minibatch_cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel("损失值")
    plt.xlabel("迭代次数")
    plt.title("损失变化效果")
    plt.show()

    # 计算预测值和测试准确率
    Z3 = forward_propagation(X_test, parameters)
    correct_prediction = tf.equal(tf.argmax(Z3, 1), tf.argmax(Y_test, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(f"在测试集上运行的准确率：{accuracy.numpy()}")

    return parameters


if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = load_dataset()
    parameters = model(X_train, Y_train, X_test, Y_test)
