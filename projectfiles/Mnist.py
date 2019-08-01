from __future__ import division
import gzip
import matplotlib.pyplot as plt
import numpy as np
import random
import math

def Randomize_Weights(weight_vector: np.ndarray):
    rand_weights = weight_vector
    for j in range(0,len(weight_vector)):
            rand_weights[j][::1] = float(random.randint(-100, 100) / 100)
    return rand_weights

def Forward_Pass(data: np.ndarray, weights, biases): # a Single Forward Pass, returns a vector output after the softmax
    output = [0] * 10
    probs = np.zeros(10)
    output[::1] = (np.inner(weights[::1], data) + biases[::1])
    sum = 0
    for index,x in enumerate(output):
        exp = math.exp(x)
        sum += exp
        probs[index] = exp
    probs[::1] = probs[::1] * (1/sum)
    return probs


def Label_Probs(probabilities): # returns a label, expects normalized data
    return np.unravel_index(np.argmax(probabilities, axis=None), probabilities.shape)

def Test_NN(weights, biases):  # Tests network and prints accuracy
    test_im = np.zeros((10000, 784))
    test_lb = None
    with open('./data/test_images.gz', 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        tmp = bytestream.read()
        tmp = np.frombuffer(tmp, dtype=np.dtype('b'), offset=16)
        tmp = np.reshape(tmp, (10000, 784))
        test_im[:, 0:783] = (tmp[0:10000, 0:783] / 128) // 1
        test_im.astype(int)
    with open('./data/test_labels.gz', 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        tmp = bytestream.read()
        test_lb = np.frombuffer(tmp, dtype=np.dtype('b'), offset=8)

    num_correct = 0
    num_ran = 0
    for i in range(0, 10000):
        output = Label_Probs(Forward_Pass(test_im[i], weights, biases))
        if output == test_lb[i]:
            num_correct += 1
        num_ran += 1
    print("TEST RESULTS\nNUMBER CORRECT: {0}\nACCURACY: {1}%".format(num_correct, round((num_correct/num_ran)*100, 5)))

def GraphCost(cost: list):
    for x in range(len(cost)):
        plt.scatter(x, cost[x], c="black")
    plt.show()

def Train_NN(weights, biases, max_iter: int, epsilon: float, step_size:float, images, labels):  # Train network with training data
    cost = []
    iteration = -1
    while iteration < max_iter:  # each loop is a training session
        iteration += 1
        cost.append(0)
        num_wrong = 0
        for i in range(1, 10000):  # go through each image
            output = Forward_Pass(images[i], weights, biases)  # probabilities dict, each key is 0...9
            expected_label = labels[i]
            predicted_label = Label_Probs(output)
            if expected_label != predicted_label:
                num_wrong += 1
            for j in range(9):
                if expected_label == j:  # we want to maximize this if true
                    weights[j, 0:784] = weights[j, 0:784] - step_size * (images[i, 0:784] * -1 * (1-output[j]))
                    biases[j] = biases[j] - step_size * (-1 * (1-output[j]))
                else:  # minimize this
                    weights[j, 0:784] = weights[j, 0:784] - step_size * (images[i, 0:784] * output[j])
                    biases[j] = biases[j] - step_size * (1 * output[j])
            cost[iteration] = cost[iteration] - math.log(output[expected_label])
        if cost[iteration - 1] < cost[iteration]:
            step_size /= 2
            if cost[iteration] - cost[iteration - 1] <= epsilon and cost[iteration] - cost[iteration - 1] >= 0:
                break
        print("iteration: {0}/{1}   Cost:{2}  Num Wrong:{3}".format(iteration, max_iter, cost[iteration], num_wrong))
    GraphCost(cost)
    return weights, biases

if __name__ == "__main__":
    # Import data
    train_im = np.zeros((10000, 784))
    train_lb = None
    with open('./data/training_images.gz', 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        tmp = bytestream.read()
        tmp = np.frombuffer(tmp, dtype=np.dtype('b'), offset=16)
        tmp = np.reshape(tmp, (60000, 784))
        train_im[:, 0:783] = (tmp[0:10000, 0:783] / 128) // 1
        train_im.astype(int)
    with open('./data/training_labels.gz', 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        tmp = bytestream.read()
        tmp = np.frombuffer(tmp, dtype=np.dtype('b'), offset=8)
        train_lb = tmp[0:10000]

    weights = Randomize_Weights(np.zeros((10, 784)))
    biases = np.zeros(10)
    print("BEFORE TRAINING")
    Test_NN(weights, biases)
    print("TRAINING")
    weights, biases = Train_NN(weights, biases, 100, .1, .2, train_im, train_lb)
    print("AFTER TRAINING")
    Test_NN(weights, biases)
