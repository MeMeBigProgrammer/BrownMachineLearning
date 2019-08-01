import random
import numpy as np
import matplotlib.pyplot as plt

def subtractVectors(x: list, y: list):
    final = []
    for i, val in enumerate(x):
        final.append(val - y[i])
    return final


def multiplyScaler(x: list, y: float):
    final = x
    for i, val in enumerate(final):
        final[i] = val * y
    return final

def perceptron(data, labels):
    weights = [random.randint(-20, 10) / 10, random.randint(-20, 10) / 10]
    print("initial weight", weights)
    i = 0
    while True:
        flag = False
        for x, val in enumerate(data):
            if np.dot(weights, val) * labels[x] < 0:
                flag = True
                scaled = multiplyScaler(val, labels[x])
                final = subtractVectors(scaled, weights)
                weights = final
        if not flag:
            print("final weight", weights)
            break
    return weights


def decision_bd(w):
    ratio = (w[0]/-w[1])
    final = [
        [-2,2],
        [ratio*-2, ratio*2]
    ]
    return final

def plot_final(data: list, labels: list, line: list, vector: list):
    i = 0
    for mat in data:
        if labels[i] == 1:
            plt.scatter(mat[0], mat[1], c="green")
        else:
            plt.scatter(mat[0], mat[1], c="red")
        i += 1
    plt.plot(line[0], line[1], 'k-')
    ratio = vector[1]/vector[0]
    plt.arrow(0,0, 1.5,ratio*1.5, fc="k", ec="k", head_width=0.15, head_length=0.1, linestyle='dotted')
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    plt.show()


if __name__ == "__main__":
    # Generate data points
    data = np.random.random((100, 2))
    data[0:50, :] = -data[0:50, :]

    # Generate labels
    labels = np.ones((100))
    labels[0:50] = -1

    # Plot data
    plt.scatter(data[0:50, 0], data[0:50, 1])
    plt.scatter(data[51:99, 0], data[51:99, 1])

    weights = perceptron(data,labels)

    decision_line = decision_bd(weights)

    plot_final(data,labels, decision_line, weights)

    # note: the vectors and decision boundaries don't look perpendicular due to the size of the graph