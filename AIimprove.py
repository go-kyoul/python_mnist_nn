import sys
import math
import time
import matplotlib.pylab as plt
import numpy as np
from dataset.mnist import load_mnist
sys.path.append("./dataset")
np.set_printoptions(precision=3)
np.set_printoptions(threshold=sys.maxsize)


#######################################################################################################
# read data

def mnist_show(n, train):
    if train:
        image = train_image_data[n]
        label = train_label_data[n]
        plt.figure(figsize=(4, 4))
        plt.title("train Image %d : label " % n + str(label))
    else:
        image = test_image_data[n]
        label = test_label_data[n]
        plt.figure(figsize=(4, 4))
        plt.title("test Image %d : label " % n + str(label))
    image_reshaped = image.reshape(28, 28)
    plt.imshow(image_reshaped, cmap="gray")
    plt.show()


def show(image):
    plt.figure(figsize=(4, 4))
    plt.title("Image")
    image_reshaped = image.reshape(28, 28)
    plt.imshow(image_reshaped, cmap="gray")
    plt.show()


(train_image_data, train_label_data), (test_image_data, test_label_data) \
    = load_mnist(flatten=True, normalize=True)
print(train_image_data.shape)
print(train_label_data.shape)
print(test_image_data.shape)
print(test_label_data.shape)
# mnist_show(0,train=1)


########################################################################################################
# math

def ReLU(x):  # array in/out
    return np.where(x > 0, x, x * 0.01)


def dReLU(x):  # array in/out
    return np.where(x > 0, 1, 0.01)


def softmax(x): # array in/out
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


def cross_entropy_error(x, label):  # array in/out
    delta = 1e-7
    return -1 * np.sum(label * np.log(x + delta))


#########################################################################################################
# layer

class Layer:
    Rate = 0.001
    Loss = 0.00005
    a = 0.9

    def __init__(self, in_size, out_size, layer_type='hidden'):
        self.layer_type = layer_type
        self.in_size = in_size
        self.out_size = out_size

        self.pre_node = np.zeros(out_size)
        self.node = np.zeros(out_size)
        self.weight = np.array(np.random.randn(in_size, out_size) / math.sqrt(in_size/2))  # He init
        self.bias = np.zeros(out_size)

        self.d = np.zeros(out_size)
        self.dw = np.zeros((in_size, out_size))
        self.db = np.zeros(out_size)
        self.v = np.zeros((in_size, out_size))

        if self.layer_type == 'output':
            self.softmax_output = np.zeros(out_size)
            self.loss = 0

    def __rshift__(self, output):  # propagation
        output.pre_node = self.node @ output.weight
        output.node = ReLU(output.pre_node) + output.bias
        if output.layer_type == 'output':
            output.softmax_output = softmax(output.node)
            output.loss = cross_entropy_error(output.softmax_output, input_label)

    def __lshift__(self, output):  # backpropagation
        if self.layer_type == 'input':
            self.node = output
            return

        if output.layer_type == 'output':
            output.d = output.softmax_output - input_label

        self.d += dReLU(self.pre_node) * (output.d @ output.weight.T)
        output.dw += self.node.reshape(1, self.out_size).T @ output.d.reshape(1, output.out_size)
        output.db += output.d

    def apply(self):
        self.v = self.a * self.v + self.Rate * (self.dw - self.a * self.v)  # NAG
        self.weight -= self.v
        self.bias -= self.Rate * self.db
        self.d = np.zeros(self.d.shape)
        self.dw = np.zeros(self.dw.shape)
        self.db = np.zeros(self.db.shape)


def test(Test_data_num):
    global input_label
    err_num = 0
    for i in range(Test_data_num):
        input_image = np.array(test_image_data[i])
        input_label = np.array([1 if x == test_label_data[i] else 0 for x in range(10)])

        L[0] << input_image
        for j in range(3):
            L[j] >> L[j + 1]

        if list(L[3].softmax_output).index(max(L[3].softmax_output)) != test_label_data[i]:
            err_num += 1
    return err_num


##########################################################################################################

L = []
L.append(Layer(784, 784, 'input'))  # 0
L.append(Layer(784, 200))  # 1
L.append(Layer(200, 100))  # 2
L.append(Layer(100, 10, 'output'))  # 3

Train_data_num = 60000
Test_data_num = 5000
Epoch = 1
Batch = 10
#Layer.Rate /= Batch


print(f"\nTest start... \nnum: {Test_data_num}")
err_num = test(Test_data_num)
print("Total Err:",err_num, "\nErr:",err_num / Test_data_num * 100,"%")


print(f"\nTrain start...\nnum: {Train_data_num}\n")
start_time = time.time()

for n in range(Epoch):
    for i in range(Train_data_num):
        input_image = np.array(train_image_data[i])
        input_label = np.array([1 if x == train_label_data[i] else 0 for x in range(10)])

        L[0] << input_image
        for j in range(len(L)-1):
            L[j] >> L[j+1]
        for j in range(len(L)-1, 0, -1):
            L[j-1] << L[j]

        if i % Batch == 0 or i == Train_data_num-1:
            for j in range(1, len(L)):
                L[j].apply()

        if (i+1) % (Epoch*Train_data_num/20) == 0:
            print(f"{round((n*Train_data_num+i)/(Epoch*Train_data_num)*100)}%")
            err_num = test(Test_data_num)
            print("Total Err:", err_num, "\nErr:", round(err_num / Test_data_num * 100, 2), "%")

print("#Time:", round(time.time() - start_time, 2), "sec")


print(f"\nTest start... \nnum: {Test_data_num}")
err_num = test(Test_data_num)
print("Total Err:",err_num, "\nErr:",err_num / Test_data_num * 100,"%")





