#Cost function in LR- MSE- diff between expected and real, min of
#this function therefore calculate gradient wrt w and b
import numpy as np
class LinearRegression:
    def __init__(self,lr=0.001,n_iters=1000):
        self.lr=lr
        self.n_iters=n_iters
        self.weights=None
        self.bias=None

    def fit(self,X,y): #training, gradient descent
        #need some initialisatin
        #init parameters
        n_samples,n_features=X.shape
        #initialise weights with 0s
        self.weights=np.zeros(n_features) #n_features is size
        self.bias=0

        for _ in range(self.n_iters):
            y_predicted=np.dot(X,self.weights)+ self.bias

            dw=(1/n_samples) * np.dot(X.T,(y_predicted-y))
            db=(1/n_samples)* np.sum(y_predicted-y)

            self.weights-=self.lr*dw
            self.bias-=self.lr*db


    def predict(self,X): #testsamples
        #approximate the values with the formula
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted

