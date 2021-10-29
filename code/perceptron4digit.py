import numpy as np
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

def sigmoid(z):
    s = 1 / (1 + np.exp(-z)) 
    return s   

def model(x_train, y_train, lr = 0.5, iters = 50):
    w = np.random.rand(x_train.shape[1], 10)
    print('start y_train', y_train.shape)
    for iter in range(iters):
        error = 0
        for i in range(y_train.shape[0]):
            temp = np.squeeze(np.dot(x_train[i], w)) 
            idx_temp = np.argmax(temp)  
            if( idx_temp != y_train[i]):
                w[:,y_train[i]] += lr*x_train[i,y_train[i]]
                error += 1
            else:
                pass
        if(error == 0):
            break
    return w

def predict(w, x_test):
    temp =np.dot(x_test, w)
    y_pred = np.zeros(temp.shape[0])
    w = w.reshape(x_test.shape[1],10)
    for i in range(x_test.shape[0]):                                  
        idx_max = np.argmax(temp[i])
        y_pred[i] = idx_max
    return y_pred

def plot(var, title, color, ylabel):
    x = np.arange(0.1, 1.1, 0.1)
    plt.plot(x, var, label = 'time', color=color)
    plt.xlabel('Percentage of Training Data')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

def process_data(data_file, label_file):
    label, sample_num = load_label(label_file)
    data = load_sample(data_file, sample_num)
    new_data=[]
    for i in range(len(data)):
        new_data.append(data[i].flatten())
    idx = np.random.shuffle(np.arange(int(len(new_data))))
    return np.squeeze(np.array(new_data)[idx]), np.squeeze(np.array(label)[idx])

def acc(pred, label):
    count=0
    print(pred.shape,label.shape)
    for i in range(pred.shape[0]):
        if(pred[i]!=label[i]):
            count+=1
    acc = count/pred.shape[0]
    return acc

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
        print('amount',amount*(i+1))
        w = model(x_train[0:amount*(i+1)],y_train[0:amount*(i+1)])
        end = time.time()
        y_pred_test = predict(w, x_test)
        accuracy = acc(np.squeeze(y_pred_test), y_test)
        print("test accuracy:{}".format(accuracy))
        time_consume.append(end-start)
        test_acc.append(accuracy)
    plot(time_consume, title='DigitImage', color='blue', ylabel="Time(s)")
    plot(test_acc, title='DigitImage', color='red', ylabel='ACC')
main()


