import numpy as np
import cv2
import scipy
from scipy import ndimage
from sklearn.utils import shuffle
from ds import load_dataset
import warnings
warnings.filterwarnings("ignore")
class Perceptron():
    def __init__(self,x_train,y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.w = np.zeros((len(self.x_train),1))
        print('w')
        print(self.w.shape)
        self.b =0
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    def propagate(self, X, Y, w,b):
        m = X.shape[1]
        A = self.sigmoid(np.dot(X.T,w) + b)          
        cost = -(1/m) * np.sum(Y.T * np.log(A) + (1 - Y.T) * (np.log(1-A)) )                                 # compute cost
        dw = (1/m) * np.dot(X,(A-Y.T))

        db = (1/m) * np.sum(A-Y.T)
        cost = np.squeeze(cost)    
        return dw, db,cost
    def optimize(self,learningRate=0.005,steps=2000):
        X = self.x_train
        Y = self.y_train
        w = self.w
        b = self.b
        costs =[]
        for i in range(steps):
            dw,db,cost =self.propagate(X,Y,w,b)
            w = w - learningRate*dw
            b = b - learningRate*db
            if i%100 ==0:
                costs.append(cost)
                print('cost after %i: %f' %(i,cost))
        print('bat dau save ...')
        self.save(w,b)
        print('saved')
        return w,b
    def load(self):
        w = np.load('weights.npy')
        b = np.load('biases.npy')
        return w,b
    def save(self,w,b):
        np.save('weights.npy',w)
        np.save('biases.npy',b)
    def predict(self,image):
        w,b = self.optimize()
        print('w')
        print(w.shape)
        m = image.shape[1]
        # w = w.reshape((1,image.shape[0]))
        Y_prediction = np.zeros((1,m))
        A = self.sigmoid(np.dot(w.T,image)+b)
        print('a')
        print(A.shape)
        for i in range(A.shape[0]):
            if A[0,i] >0.5:
                Y_prediction[0,i] = 1
            else:
                Y_prediction[0,i] =0
        print(Y_prediction)
        return Y_prediction
    

# x_train,y_train = [],[]
# for i in range(10):
#     img = cv2.imread('C:\\Users\\Hi-XV\\Desktop\\dogs-vs-cats-redux-kernels-edition\\train\\cat.' + str(i) + '.jpg')
#     img = cv2.resize(img,(64,64))
#     x_train.append(img)
#     y_train.append(0)
#     img2 = cv2.imread('C:\\Users\\Hi-XV\\Desktop\\dogs-vs-cats-redux-kernels-edition\\train\\dog.' + str(i) + '.jpg')
#     img2 = cv2.resize(img,(64,64))
#     x_train.append(img2)
#     y_train.append(1)
# x_train = np.array(x_train)
# num_px = x_train.shape[1]
# y_train = np.array(y_train)
# x_train, y_train = shuffle(x_train, y_train, random_state = 0)
# y_train = y_train.reshape(-1,1)
# x_train_flatten = x_train.reshape(x_train.shape[0], -1).T
# x_train = x_train_flatten / 255
# print('x shape')
# print(x_train.shape)
# print('y shape')
# print(y_train.shape)




x_train,y_train,x_test,y_test,classes = load_dataset()
print(y_train.shape)
# cv2.imshow('image',x_train[2])
# cv2.waitKey(0)
m_train = x_train.shape[0]
m_test = x_test.shape[0]
num_px = x_train.shape[1]
train_set_x_flatten = x_train.reshape(x_train.shape[0], -1).T
test_set_x_flatten = x_test.reshape(x_test.shape[0], -1).T
x_train = train_set_x_flatten/255.
x_test = test_set_x_flatten/255.
print('x shape')
print(x_train.shape)
print('y shape')
print(y_train.shape)





pct = Perceptron(x_train,y_train)

pd_img = np.array(ndimage.imread('C:\\Users\\Hi-XV\\Desktop\\dogs-vs-cats-redux-kernels-edition\\train\\cat.0.jpg',flatten=False))
my_image = scipy.misc.imresize(pd_img, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T

print(my_image.shape)
pct.predict(my_image)
