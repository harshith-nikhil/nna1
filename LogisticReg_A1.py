#!/usr/bin/env python
# coding: utf-8
import pandas as pd 
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Standard Sigmoid Function
def sigmoid(z):
    S_z=1/(1+np.exp(-z))
    return S_z

def set_params(n_f):
    w=np.ones(n_f)
    b=0
    return w,b

#Computing the Total Average Loss
def cost_function(y, y_hat, m):
    J_w_b = -(1 / m) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    return J_w_b

# Gradient Descent
def update_params(X, y, y_hat, m):
    dw = (1 / m) * np.dot((y_hat - y).T, X)
    db = (1 / m) * np.sum(y_hat - y)
    return dw, db

# Visualize the Loss Over Epochs
def plot_loss(loss_over_epochs):
    plt.figure(figsize=(8,6))
    plt.plot(range(len(loss_over_epochs)), loss_over_epochs, label='Loss Reduction Over epochs', color='orange')
    plt.title('Loss Reduction Over epochs')
    plt.show()

def make_preds(X, w, b):
    z = np.dot(X, w) + b
    y_hat = sigmoid(z)
    return [1 if i >= 0.5 else 0 for i in y_hat]

# Fit the Model
def model_fit(X, y, alpha, epochs):
    m, n = X.shape
    w, b = set_params(n)
    loss_over_epochs = []
    for i in range(epochs):
        z = np.dot(X, w) + b
        y_hat = sigmoid(z)
        J = cost_function(y, y_hat, m)
        loss_over_epochs.append(J)
        dw, db = update_params(X, y, y_hat, m)
        w -= alpha * dw
        b -= alpha * db
        if i % 50 ==0:
            print(f"Epoch {i}, Cost: {J}")
    return w, b, loss_over_epochs

def plot_confusion_matrix(y_true, y_hat):
    cm = confusion_matrix(y_hat, y_true)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False,xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.title("Confusion Matrix")
    plt.show()

data = pd.read_csv('A1_Data_VenkataHarshithNikhil.csv')
X = data[['years_since','product_satisfaction','service_satisfaction']]
y = data['retained']

alpha = 0.01
epochs = 500
w, b, loss_over_epochs = model_fit(X, y, alpha, epochs)

plot_loss(loss_over_epochs)

X_test = X.to_numpy()
y_true = y.to_numpy()
y_hat = make_preds(X_test, w, b)

plot_confusion_matrix(y_true, y_hat)