
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from ds import load_dataset
from sklearn.model_selection import train_test_split
import warnings
import cv2
from sklearn.utils import shuffle
warnings.filterwarnings('ignore')



class Perceptron():
    def __init__(self,x_train,y_train,x_test,y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
    def sigmoid(self,z): 
        s = 1 / (1 + np.exp(-z))
        return s
    def initialize_with_zeros(self,dim):  
        w = np.zeros((dim, 1))
        b = 0
        assert(w.shape == (dim, 1))
        assert(isinstance(b, float) or isinstance(b, int))
        return w, b
    def propagate(self,w, b, X, Y):
        m = X.shape[1]
        A = self.sigmoid(np.dot(X.T,w) + b)          
        cost = -(1/m) * np.sum(Y.T * np.log(A) + (1 - Y.T) * (np.log(1-A)) )                                 
        dw = (1/m) * np.dot(X,(A-Y.T))
        db = (1/m) * np.sum(A-Y.T)
        assert(dw.shape == w.shape)
        assert(db.dtype == float)
        cost = np.squeeze(cost)
        assert(cost.shape == ())
        
        grads = {"dw": dw,
                "db": db}
        
        return grads, cost
    def optimize(self,w, b, X, Y, num_iterations, learning_rate, print_cost = False):
        costs = []
        
        for i in range(num_iterations):

            grads, cost = self.propagate(w, b, X, Y)
            dw = grads["dw"]
            db = grads["db"]
            w = w - learning_rate * dw
            b = b - learning_rate * db      
            if i % 100 == 0:
                costs.append(cost)    
            if print_cost and i % 100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))   
        params = {"w": w,
                "b": b}
        
        grads = {"dw": dw,
                "db": db}
        print('saving..')
        self.save(w,b)
        print('saved')
        return params, grads, costs
    def save(self,w,b):
        np.save('weights.npy',w)
        np.save('biases.npy',b)
    def load(self):
        w = np.load('weights.npy')
        b = np.load('biases.npy')
        return w,b
    def predict(self, X):  
        w,b = self.load()
        m = X.shape[1]
        Y_prediction = np.zeros((1,m))
        w = w.reshape(X.shape[0], 1) 
        A = self.sigmoid(np.dot(X.T,w) + b)
        
        print(range(A.shape[1]))
        for i in range(A.shape[0]):            
            if(A[i,0] > 0.5):
                Y_prediction[0,i] = 1
            else:
                Y_prediction[0,i] = 0      
        assert(Y_prediction.shape == (1, m))
        
        return Y_prediction

    def model(self, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
        X_train, Y_train, X_test, Y_test = self.x_train,self.y_train,self.x_test,self.y_test
        w, b = self.initialize_with_zeros(X_train.shape[0])
        
        parameters, grads, costs = self.optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
        w = parameters["w"]
        b = parameters["b"]
        Y_prediction_test = self.predict( X_test)
        Y_prediction_train = self.predict( X_train)
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

        
        d = {"costs": costs,
            "Y_prediction_test": Y_prediction_test, 
            "Y_prediction_train" : Y_prediction_train, 
            "w" : w, 
            "b" : b,
            "learning_rate" : learning_rate,
            "num_iterations": num_iterations}
        
        return d
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.
pct = Perceptron(train_set_x, train_set_y, test_set_x, test_set_y)
pct.model()
image = np.array(ndimage.imread('C:\\Users\\Hi-XV\\Desktop\\dogs-vs-cats-redux-kernels-edition\\test\\167.jpg', flatten=False))
my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
print(pct.predict(my_image))

plt.imshow(image)
