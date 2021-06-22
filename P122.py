import cv2
import numpy as np
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import accuracy_score as acs
X = np.load('image.npz')['arr_0']
y = pd.read_csv('P122.csv')["labels"]
print(pd.Series(y).value_counts())
classes = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']
nclasses= len(classes)
samples_per_class = 5
figure = plt.figure(figsize=(nclasses*2,(1+ samples_per_class *2)))
idx_cls = 0

for cls in classes:
    idxs = np.flatnonzero(y == cls)
    idxs = np.random.choice(idxs, samples_per_class, replace = False)
    i = 0
    for idx in idxs:
        plt_idx = i *nclasses + idx_cls + 1
        p = plt.subplot(samples_per_class, nclasses, plt_idx);
        p = sns.heatmap(np.reshape(X[idx], (22,30)), cmap=plt.cm.gray, xticklabels = False, yticklabels = False, cbar = False);
        p = plt.axis('off')
        i+=1
    idx_cls +=1
x_train, x_test, y_train, y_test = tts(X, y, random_state = 9, train_size = 7500, test_size = 2500)
xtrain = x_train/255.0
xtest = x_test/255.0

model = lr(solver = 'saga', multi_class='multinomial').fit(xtrain, y_train)

modeltest = model.predict(xtest)
accuracy = acs(y_test, modeltest)
print(accuracy)


metrics = pd.crosstab(y_test, modeltest, rownames=['Actual'], colnames=['Prediction'])
fig = mp.figure(figsize=(10,10))
fig = sns.heatmap(metrics, annot=True, fmt="d", cbar=False)
