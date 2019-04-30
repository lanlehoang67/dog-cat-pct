
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


# train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
# m_train = train_set_x_orig.shape[0]
# m_test = test_set_x_orig.shape[0]
# num_px = train_set_x_orig.shape[1]
# train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
# test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
# train_set_x = train_set_x_flatten/255.
# test_set_x = test_set_x_flatten/255.

x_train,y_train = [],[]
for i in range(10):
    img = cv2.imread('C:\\Users\\Hi-XV\\Desktop\\dogs-vs-cats-redux-kernels-edition\\train\\cat.' + str(i) + '.jpg')
    img = cv2.resize(img,(64,64))
    x_train.append(img)
    y_train.append(0)
    img2 = cv2.imread('C:\\Users\\Hi-XV\\Desktop\\dogs-vs-cats-redux-kernels-edition\\train\\dog.' + str(i) + '.jpg')
    img2 = cv2.resize(img,(64,64))
    x_train.append(img2)
    y_train.append(1)
x_train = np.array(x_train)
num_px = x_train.shape[1]
y_train = np.array(y_train)
# y_train = y_train.reshape(1,-1)


x_train, y_train = shuffle(x_train, y_train, random_state = 0)
train_set_x, test_set_x, train_set_y, test_set_y = train_test_split(x_train, y_train, test_size = 0.2, random_state = 0)
x_train_flatten =train_set_x.reshape(train_set_x.shape[0], -1).T
test_set_x_flatten = test_set_x.reshape(test_set_x.shape[0],-1).T
train_set_x = x_train_flatten / 255.
test_set_x = test_set_x_flatten/255.
train_set_y = train_set_y.reshape(1,len(train_set_y))
test_set_y = test_set_y.reshape(1,len(test_set_y))





print('train x',train_set_x.shape)
print('test x',test_set_x.shape)
print('train y',train_set_y.shape)
print('test y',test_set_y.shape)
def sigmoid(z): 
    s = 1 / (1 + np.exp(-z))
    return s
def initialize_with_zeros(dim):  
    w = np.zeros((dim, 1))
    b = 0
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b
dim = 2
w, b = initialize_with_zeros(dim)

def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(X.T,w) + b)          
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
w, b, X, Y = np.array([[1],[2]]), 2, np.array([[1,2],[3,4]]), np.array([[1,0]])
grads, cost = propagate(w, b, X, Y)

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = []
    
    for i in range(num_iterations):

        grads, cost = propagate(w, b, X, Y)
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
    
    return params, grads, costs
params, grads, costs = optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)


def predict(w, b, X):  
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1) 
    A = sigmoid(np.dot(X.T,w) + b)
    
    print(range(A.shape[1]))
    for i in range(A.shape[0]):            
        if(A[i,0] > 0.5):
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0      
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    w, b = initialize_with_zeros(X_train.shape[0])
    print(w.shape)
    
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = parameters["w"]
    b = parameters["b"]
    Y_prediction_test = predict(w,b, X_test)
    Y_prediction_train = predict(w,b, X_train)
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
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

image = np.array(ndimage.imread('C:\\Users\\Hi-XV\\Cats-vs-Dogs-Classification-CNN-Keras-\\dog-or-cat.jpg', flatten=False))
my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
print(predict(d["w"], d["b"], my_image))

plt.imshow(image)
