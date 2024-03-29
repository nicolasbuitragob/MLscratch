import numpy as np
import matplotlib.pyplot as plt


class Classification():
    def __init__(self):
        self._iterations = 2000
        self._learning_rate = 0.001
        self.theta = None
        self.cost = []

    def sigmoid(self,x):
        z = 1/(1 + np.exp(-x))
        return z

    def fit(self,X,y):
        m,n = X.shape
        self.theta = np.random.randn(n) 
        X_with_bias = np.c_[np.ones(m),X] 
        self.theta = np.insert(self.theta,0,0)
        for i in range(self._iterations):
            h = np.dot(X_with_bias,self.theta)
            z = self.sigmoid(h)    
            J = (-1/m)*np.sum((y*np.log(z))+((1-y)*np.log(1-z)))
            error = z - y
            grad = (1/m)*np.dot(X_with_bias.T,error)
            self.theta =  self.theta - self._learning_rate*grad
            self.cost.append(J)
        self.plot()

    def predict(self, X_test):        
        m_test = X_test.shape[0]
        X_test_bias = np.c_[np.ones(m_test),X_test]
        pred = np.dot(X_test_bias,self.theta)
        return (pred >= 0.5 )*1