import matplotlib.pyplot as plt
import numpy as np
import random

def g(x):
    return x ** 4 + x ** 3 + 2 * x ** 2 + 2 * x


def g_prime(x):
    return (4*x**3) + (3*x**2) + (4*x) + 2


def printStatus(iteration: int, weight: float):
  print("=================")
  print("Iteration: {}".format(iteration))
  print("Weight: {}".format(weight))
  print("Derivative Value of Weight: {}".format(g_prime(weight)))
  print("=================")


def plotFuncAndWgt(weight: float):
  ## make plot ##
  plt.scatter(weight, g(weight), c="green")
  x = np.arange(-10, 10, 0.1)
  plt.plot(x,g(x), "r")
  plt.xlim(-5, 5)
  plt.ylim(-3, 5)
  plt.show()


# Gradient descent
STUFF = []
max_iter = 100  # you choose
step_size = .02  # you choose
epsilon = .001  # when the difference is so small
conv = False
weight = float(random.randint(-500, 500)/100.0)  # choose starting point
last_wt = weight
i = 0

while i < max_iter and not conv:
  try:
    last_wt = weight
    plt.scatter(weight, g(weight))
    weight = weight - (step_size*g_prime(weight))
  except OverflowError:
    step_size *= .5

  if g(weight) > g(last_wt):
    step_size *= .5
  if abs(g_prime(last_wt) - g_prime(weight)) < epsilon:
    break
  elif g_prime(weight) == 0:
    conv = True

  if i % 50 == 0:
    printStatus(i, weight)
  STUFF.append([i,g_prime(weight)])
  i += 1

printStatus(i, weight)
plotFuncAndWgt(weight)


