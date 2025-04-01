import sys
import math
import time
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import random
import PIL.Image as Pil
from dataset.mnist import load_mnist
sys.path.append("./dataset")
np.set_printoptions(precision=3)
np.set_printoptions(threshold=sys.maxsize)


#######################################################################################################
# read data

Image_height = 28
Image_width = 28


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
    image_reshaped = image.reshape(Image_height, Image_width)
    plt.imshow(image_reshaped, cmap="gray")
    plt.show()


def show(image):
    plt.figure(figsize=(4, 4))
    plt.title("Image")
    image_reshaped = image.reshape(Image_height, Image_width)
    plt.imshow(image_reshaped, cmap="gray")
    plt.show(block=False)


def text_show(image):
    for i in range(Image_height):
        for j in range(Image_width):
            if image[i][j] > 200/255:
                print('@', end=" ")
            elif image[i][j] > 80/255:
                print('+', end=" ")
            elif image[i][j] > 1/255:
                print('*', end=" ")
            else:
                print('.', end=" ")
        print("")


(train_image_data, train_label_data), (test_image_data, test_label_data) \
    = load_mnist(flatten=True, normalize=True)
print(train_image_data.shape)
print(train_label_data.shape)
print(test_image_data.shape)
print(test_label_data.shape)
test_err_list = []
train_err_list = []


########################################################################################################
# math

def ReLU(x):  # array in/out
    return np.where(x > 0, x, x * 0.01)


def dReLU(x):  # array in/out
    return np.where(x > 0, 1, 0.01)


def softmax(x):  # array in/out
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


def cross_entropy_error(x, label):  # array in/out
    delta = 1e-7
    return -1 * np.sum(label * np.log(x + delta))


#########################################################################################################
# layer

class Layer:
    Rate = 0.002
    Loss = 0.0001
    a = 0.9

    def __init__(self, in_size, out_size, layer_type='hidden'):
        self.layer_type = layer_type
        self.in_size = in_size
        self.out_size = out_size

        self.label = np.zeros(Output_data_size)
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
            output.loss = cross_entropy_error(output.softmax_output, L[0].label)

    def __lshift__(self, output):  # backpropagation
        if self.layer_type == 'input':
            if np.size(output) == Input_data_size:
                self.node = output
            elif np.size(output) == Output_data_size:
                self.label = output
            return

        if output.layer_type == 'output':
            output.d = output.softmax_output - L[0].label

        self.d += dReLU(self.pre_node) * (output.d @ output.weight.T)
        output.dw += self.node.reshape(1, self.out_size).T @ output.d.reshape(1, output.out_size)
        output.db += output.d

    def apply(self):
        self.v = self.a * self.v + self.Rate * (self.dw - self.a * self.v)  # NAG
        self.weight -= self.v
        #  self.weight -= self.Rate * (self.dw + self.Loss*self.weight)
        self.bias -= self.Rate * self.db
        self.d = np.zeros(self.d.shape)
        self.dw = np.zeros(self.dw.shape)
        self.db = np.zeros(self.db.shape)


def run(layer):
    for n in range(Epoch):
        start, end = Train_data_limit
        temp = random.sample(range(start, end), Train_data_num)
        for i in range(Train_data_num):
            input_image = np.array(train_image_data[temp[i]])
            input_label = np.array([1 if x == train_label_data[temp[i]] else 0 for x in range(Output_data_size)])

            layer[0] << input_image
            layer[0] << input_label
            for j in range(len(layer) - 1):
                layer[j] >> layer[j + 1]
            for j in range(len(layer) - 1, 0, -1):
                layer[j - 1] << layer[j]

            if i % Batch == 0 or i == Train_data_num - 1:
                for j in range(1, len(layer)):
                    layer[j].apply()

            if (n*Train_data_num + i+1) % int(Epoch * Train_data_num / 20) == 0:
                print(f"{round((n * Train_data_num + i) / (Epoch * Train_data_num) * 100)}%")
                if testdata_test_during_train:
                    calc_err(test(L), 'test')
                if traindata_test_during_train:
                    calc_err(train_test(L), 'train')


def test(layer):
    err_cnt = 0
    for i in range(Test_data_num):
        input_image = np.array(test_image_data[i])
        input_label = np.array([1 if x == test_label_data[i] else 0 for x in range(Output_data_size)])

        layer[0] << input_image
        layer[0] << input_label
        for j in range(len(layer)-1):
            layer[j] >> layer[j + 1]

        if list(layer[len(layer)-1].softmax_output).index(max(layer[len(layer)-1].softmax_output)) \
                != test_label_data[i]:
            err_cnt += 1
    return err_cnt


def train_test(layer):
    start, end = Train_data_limit
    temp = random.sample(range(start, end), Test_data_num)
    train_err_cnt = 0
    for i in range(Test_data_num):
        input_image = np.array(train_image_data[temp[i]])
        input_label = np.array([1 if x == train_label_data[temp[i]] else 0 for x in range(Output_data_size)])

        layer[0] << input_image
        layer[0] << input_label
        for j in range(len(layer) - 1):
            layer[j] >> layer[j + 1]

        if list(layer[len(layer) - 1].softmax_output).index(max(layer[len(layer) - 1].softmax_output)) != \
                train_label_data[temp[i]]:
            train_err_cnt += 1
    return train_err_cnt


def calc_err(err_num, type):
    if type == 'test':
        test_err_list.append(round(err_num / Test_data_num * 100, 2))
        print("Total test Err:", err_num, "\nErr:", round(err_num / Test_data_num * 100, 2), "%")
    else:
        train_err_list.append(round(err_num / Test_data_num * 100, 2))
        print("Total train Err:", err_num, "\nErr:", round(err_num / Test_data_num * 100, 2), "%")


def user_test(layer):
    chk = input("\nYour Input: ")
    while 1:
        user_test_image_temp = np.array(Pil.open('file.bmp'))
        user_test_image = np.zeros((Image_height, Image_width))
        for i in range(Image_height):
            for j in range(Image_width):
                user_test_image[i][j] = user_test_image_temp[i][j][1] / 255
        text_show(user_test_image)
        input_label = np.array([1 if str(x) == chk else 0 for x in range(Output_data_size)])

        layer[0] << user_test_image.reshape(Input_data_size)
        layer[0] << input_label
        for j in range(len(layer) - 1):
            layer[j] >> layer[j + 1]

        print("AI Output:", list(layer[len(layer) - 1].softmax_output).index(\
            max(list(layer[len(layer) - 1].softmax_output))))
        print([round(x, 2) for x in list(layer[len(layer) - 1].softmax_output)], "\nLoss:", layer[len(layer) - 1].loss)

        chk = input("\nYour Input: ")
        if chk == '-1':
            break


##########################################################################################################
# main

Input_data_size = Image_height * Image_width  # 28*28
Output_data_size = 10

L = [Layer(28*28, 28*28, 'input'),
     Layer(28*28, 500),
     Layer(500, 300),
     Layer(300, 10, 'output')]  # create layer

Train_data_limit = (0, 60000)
Train_data_num = 60000
Test_data_num = 5000
Epoch = 5
Batch = 10
testdata_test_during_train = False
traindata_test_during_train = False


print(f"\nTest start... \nnum: {Test_data_num}")
calc_err(test(L), 'test')
test_err_list.pop()

print(f"\nTrain start...\nlimit: {Train_data_limit}\nnum: {Train_data_num}\nepoch: {Epoch}\nbatch: {Batch}\n")
start_time = time.time()
run(L)
print("#Time:", round(time.time() - start_time, 2), "sec")

print(f"\nTest start... \nnum: {Test_data_num}")
calc_err(test(L), 'test')
test_err_list.pop()


if testdata_test_during_train and traindata_test_during_train:
    print(len(train_err_list), len(test_err_list))
    raw_data = {'train Err': train_err_list,
                'test Err': test_err_list}
    raw_data = pd.DataFrame(raw_data)
    raw_data.to_excel(excel_writer='mnist.xlsx')


print("\nUser test start...")
user_test(L)
