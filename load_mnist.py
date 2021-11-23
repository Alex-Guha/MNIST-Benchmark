import numpy as np
import cv2, os, pickle

def load_mnist_dataset(dataset, path, channels=None):
	IMG_SIZE = 28
	labels = os.listdir(os.path.join(path, dataset))
	X = []
	y = []
	for label in labels:
		for file in os.listdir(os.path.join(path, dataset, label)):
			image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)
			if not channels:
				X.append(image)
			else:
				img = np.array(cv2.resize(image, (IMG_SIZE, IMG_SIZE)))
				X.append([channels, img[0], img[1]])
			y.append(label)
	return np.array(X), np.array(y).astype('uint8')

def create_data_mnist(path, channels=None):
	X, y = load_mnist_dataset('train', path, channels)
	X_test, y_test = load_mnist_dataset('test', path, channels)
	return X, y, X_test, y_test

def save_data(path, X, y, X_test, y_test):
	with open(path, 'wb') as f:
			pickle.dump([X, y, X_test, y_test], f)

def load_data(path):
	with open(path, 'rb') as f:
		data = pickle.load(f)
	return data[0], data[1], data[2], data[3]