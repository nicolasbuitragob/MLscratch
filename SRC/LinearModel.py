import numpy as np
import matplotlib.pyplot as plt


class Classification():
    def __init__(self):
        self._iterations = 2000
        self._learning_rate = 0.01
        self.theta = None
        self.bias = None
        self.cost = []

    def sigmoid(self,x):
        z = 1/(1 + np.exp(-x))
        return z

    def plot_graph(self):
        plt.plot(self.cost)
        plt.show()
    
    def fit(self,X,y):
        m,n = X.shape
        y = y.reshape((m,1))
        self.theta = np.random.randn(n,1)
        self.bias = 1
        for i in range(self._iterations):
            h = np.dot(X,self.theta) + self.bias
            z = self.sigmoid(h)    
            J = (-1/m)*np.sum((y*np.log(z))+((1-y)*np.log(1-z)))
            error = z - y
            grad = (1/m)*np.dot(X.T,error)
            dbias = (1/m)*np.sum(error)
            self.theta =  self.theta - self._learning_rate*grad
            self.bias = self.bias - self._learning_rate*dbias
            self.cost.append(J)
            
        self.plot_graph()
        

    def predict(self, X_test):        
        pred = self.sigmoid(np.dot(X_test, self.theta) + self.bias)
        return (pred >= 0.5 )*1