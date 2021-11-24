import numpy as np
import math
from layer import *
from activation import *
from optimizer import *
from accuracy import *
from loss import *
from model import *
from load_mnist import *
import matplotlib.pyplot as plt

X, y, X_test, y_test = load_data('D:/Datasets/fashion_mnist.pickle')

model = Model()

model.add(Layer_Dense(X.shape[1], 128))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.1))
model.add(Layer_Dense(128, 128))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.1))
model.add(Layer_Dense(128, 10))
model.add(Activation_Softmax())

# If you aren't training (aka setting the model parameters), don't need to set optimizer.
model.set(loss=Loss_CategoricalCrossEntropy(), optimizer=Optimizer_Adam(decay=5e-8), accuracy=Accuracy_Categorical())

model.finalize()
model.train(X, y, validation_data=(X_test, y_test), epochs=100, batch_size=128, print_every=10)

# Once finalized a network, save it and push it to github
model.save_parameters("100_epoch_fashMNIST_network")