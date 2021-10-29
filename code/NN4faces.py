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

def load_sample(sample_file, sample_num, pool):
    f = open(sample_file)
    line = f.readlines()
    file_length = int(len(line)) 
    width = int(len(line[0]))  
    length = int(file_length/sample_num) 
    all_image = []
    print(len(line[0]),file_length/sample_num )
    print(width, length)
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
    new_row = int(length/pool)
    new_col = int(width/pool)
    new_all_image = np.zeros((sample_num, new_row, new_col))
    for i in range(sample_num):
        for j in range(new_row):
            for k in range(new_col):
                new_pixel = 0
                for row in range(pool*j,pool*(j+1)):
                    for col in range(pool*k,pool*(k+1)):
                        new_pixel += all_image[i][row,col]
                new_all_image[i,j,k] = new_pixel
    return new_all_image

def process_data(data_file, label_file, pool):
    label, sample_num = load_label(label_file)
    data = load_sample(data_file, sample_num, pool)
    new_data=[]
    for i in range(len(data)):
        new_data.append(data[i].flatten())
    idx = np.random.shuffle(np.arange(int(len(new_data))))
    return np.squeeze(np.array(new_data)[idx]), np.squeeze(np.array(label)[idx])


def optimization(w, b, x, y, iter, lr):
    for i in range(iter):
        dw, db, cost = propagation(w, b, x, y)
        w = w - lr*dw
        b = b - lr*db
        # if i % 100 == 0:
        #     print("cost after iteration{}: {}".format(i, cost))
    return w, b, dw ,db

def propagation(w, b, x,y):
    m = x.shape[0]
    atv = np.squeeze(sigmoid(np.dot(x,w)+b)) 
    y = np.array([int(item) for item in y])
    cost = -(1/m)*np.sum(y*np.log(atv)+(1-y)*np.log(1-atv)) 
    dw = (1/m)*np.dot(x.T,(atv-y)).reshape(w.shape[0],1)
    db = (1/m)*np.sum(atv-y)
    return dw, db, cost

def sigmoid(z):
    s = 1 / (1 + np.exp(-z)) 
    return s   

def predict(w, b, x ):
    w = w.reshape(x.shape[1], 1)
    y_pred = sigmoid(np.dot(x, w) + b)
    for i in range(y_pred.shape[0]):
        if(y_pred[i] > 0.5):

            y_pred[i] = 1
        else:
            y_pred[i] = 0
    return y_pred

def model(x_train, y_train, iter = 2000, lr = 0.5):
    w = np.zeros((x_train.shape[1],1));b = 0
    w,b, dw, db = optimization(w, b, x_train, y_train, iter, lr)
    return w, b

def plot(var, title, color, ylabel):
    x = np.arange(0.1, 1.1, 0.1)
    plt.plot(x, var, label = 'time', color=color)
    plt.xlabel('Percentage of Training Data')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

def acc(pred, label):
    acc = 1 - np.mean(np.abs(pred-label))
    return acc

def main():
    pool = 3
    train = "../data/facedata/facedatatrain"
    train_label = "../data/facedata/facedatatrainlabels"
    test = "../data/facedata/facedatatest"
    test_label = "../data/facedata/facedatatestlabels"
    x_train, y_train = process_data(train, train_label, pool)
    x_test, y_test = process_data(test, test_label, pool)
    amount = int(x_train.shape[0]/10)
    time_consume = []
    test_acc = []
    for i in range(10):
        start = time.time()
        w, b = model(x_train[0:amount*(i+1)],y_train[0:amount*(i+1)])
        end = time.time()
        y_pred_test = predict(w, b, x_test)
        y_pred_train = predict(w, b, x_train)
        test_accuracy = acc(np.squeeze(y_pred_test), y_test)
        print("test accuracy:{}".format(test_accuracy))
        time_consume.append(end-start)
        test_acc.append(test_accuracy)
    print(time_consume)
    print(test_acc)
    plot(time_consume, title='FaceImage', color='blue', ylabel="Time(s)")
    plot(test_acc, title='FaceImage', color='red', ylabel='ACC')
main()