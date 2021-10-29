import numpy as np
import random
import time
import matplotlib.pyplot as plt
def load_label(label_file):
    f = open(label_file)
    line = f.readlines()
    line = [int(item.strip()) for item in line]
    sample_num = len(line)
    return line, sample_num

def load_sample(sample_file, sample_num):
    f = open(sample_file)
    line = f.readlines()
    file_length = int(len(line)) 
    width = int(len(line[0]))  
    length = int(file_length/sample_num) 
    all_image = []
    for i in range(sample_num):
        single_image = np.zeros((length,width))
        count=0
        for j in range(length*i,length*(i+1)): 
            single_line=line[j]
            for k in range(len(single_line)):
                if(single_line[k] == "+" or single_line[k] == "#"):
                    single_image[count, k] = 1 
            count+=1        
        all_image.append(single_image) 
    return all_image

def one_hot(data):
    for i in range(len(data)):
        sr = [0,0,0,0,0,0,0,0,0,0]
        sr[int(data[i])] = 1
        data[i] = np.array(sr)
    data=np.array(data)
    return data

def process_data(data_file, label_file):
    label, sample_num = load_label(label_file)
    data = load_sample(data_file, sample_num)
    label = one_hot(label)
    new_data=[]
    for i in range(len(data)):
        new_data.append(data[i].flatten())
    idx = np.random.shuffle(np.arange(int(len(new_data))))
    return np.squeeze(np.array(new_data)[idx]), np.squeeze(np.array(label)[idx])

def optimize(w, b, x, y, iter, lr):
    for i in range(iter):
        dw, db ,cost = propagation(w, b, x, y)
        w = w - lr*dw 
        b = b - lr*db
        # if i % 200 == 0:
        #     print("cost after iteration {}: {}" .format(i, cost))
    return w, b, dw, db

def propagation(w, b, x, y):
   m = x.shape[0]
   atv = np.squeeze(sigmoid(np.dot(x,w)+b))
   cost = -(1/m)*np.sum(y*np.log(atv)+(1-y)*np.log(1-atv)) 
   dw = (1/m)*np.dot(x.T,(atv-y)).reshape(w.shape[0],10)
   db = (1/m)*np.sum(atv-y)
   return dw, db, cost

def sigmoid(z): 
    s = 1 / (1 + np.exp(-z)) 
    return s   


def predict(w, b, x ):
    w = w.reshape(x.shape[1], 10)
    y_pred = sigmoid(np.dot(x, w) + b)
    for i in range(y_pred.shape[0]):
        init = [0]*10
        idx_max = np.argmax(y_pred[i]) 
        init[idx_max] = 1
        y_pred[i] = init
    return y_pred

def acc(pred, label):
    cut = pred - label
    count = 0
    for i in range(cut.shape[0]):
        if((cut[i] == 1.0).any()): 
            count += 1
    acc = 1-count/pred.shape[0]
    return acc

def model(x_train, y_train, iter = 2000, lr = 0.6):
    print('model x_train_shape', x_train.shape)
    w = np.zeros((x_train.shape[1],10));b = [0]*10
    w, b, dw, db = optimize(w, b, x_train, y_train, iter, lr)
    return w, b

def plot(var, title, color, ylabel):
    x = np.arange(0.1, 1.1, 0.1)
    plt.plot(x, var, label = 'time', color=color)
    plt.xlabel('Percentage of Training Data')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

def main():
    train = "../data/digitdata/trainingimages"
    train_label = "../data/digitdata/traininglabels"
    test = "../data/digitdata/testimages"
    test_label = "../data/digitdata/testlabels"
    x_train, y_train = process_data(train, train_label)
    x_test, y_test = process_data(test, test_label)
    amount = int(x_train.shape[0]/10)
    time_consume = []
    test_acc = []
    for i in range(10):
        start = time.time()
        w, b = model(x_train[0:amount*(i+1)],y_train[0:amount*(i+1)])
        end = time.time()
        y_pred_test = predict(w, b, x_test)
        accuracy = acc(y_pred_test, y_test)
        print("test accuracy:{}".format(accuracy))
        time_consume.append(end-start)
        test_acc.append(accuracy)
    plot(time_consume, title='DigitImage', color='blue', ylabel='Time(s)')
    plot(test_acc, title='DigitImage', color='red', ylabel='ACC')

main()

