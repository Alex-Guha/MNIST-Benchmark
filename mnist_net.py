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

model.add(Layer_Dense(X.shape[1], 64))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.1))
model.add(Layer_Dense(64, 64))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.1))
model.add(Layer_Dense(64, 10))
model.add(Activation_Softmax())

# If you aren't training (aka setting the model parameters), don't need to set optimizer.
model.set(loss=Loss_CategoricalCrossEntropy(), optimizer=Optimizer_Adam(decay=5e-5), accuracy=Accuracy_Categorical())

model.finalize()
model.train(X, y, validation_data=(X_test, y_test), epochs=8, batch_size=128, print_every=10)