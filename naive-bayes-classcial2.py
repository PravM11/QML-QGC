import numpy as np
from sklearn import datasets

data = datasets.load_iris()
features = data['data']
classes = data['target']
names = data['target_names']

n_features = features.shape[1]
n_classes = names.size

feature_means = np.zeros((n_classes, n_features))
feature_stds = np.zeros((n_classes, n_features))
for i in range(n_classes):
    for j in range(n_features):
        feature_means[i, j] = np.mean(features[classes == i, j])
        feature_stds[i, j] = np.std(features[classes == i, j])

def gaussian_prob(x, mean, std):
    return 1 / np.sqrt(2 * np.pi * std ** 2) * np.exp(-(x - mean) ** 2 / (2 * std ** 2))

def predict(x):
    y = np.ones(n_classes)
    for i in range(n_classes):
        for j in range(n_features):
            y[i] *= gaussian_prob(x[j], feature_means[i, j], feature_stds[i, j])
    return names[np.argmax(y)]

x = [5.0,3.6,1.4,0.2]
print(predict(x))
x = [6.1,2.9,4.7,1.4]
print(predict(x))
x = [6.7,2.5,5.8,1.8]
print(predict(x))