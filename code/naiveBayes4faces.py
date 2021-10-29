import numpy as np
import time
import matplotlib.pyplot as plt
def load_label(label_file):
    f = open(label_file)
    line = f.readlines()
    line = [int(item.strip()) for item in line]
    for i in range(len(line)):
        if(line[i] <= 0):
            line[i] = 0
    sample_num = len(line)
    return line, sample_num

def load_sample(sample_file, sample_num,pool):
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

def process_data(data_file, label_file,pool):
    label, sample_num = load_label(label_file)
    data = load_sample(data_file, sample_num, pool)
    new_data=[]
    for i in range(len(data)):
        new_data.append(data[i].flatten())
    idx = np.random.shuffle(np.arange(int(len(new_data))))   
    return np.squeeze(np.array(new_data)[idx]), np.squeeze(np.array(label)[idx])

def calc_prob(x_train, y_train, pool):
    sample_num = y_train.shape[0];label_num = np.unique(y_train).shape[0]; feature_nums = x_train.shape[1]; feature_values = pool*pool+1 
    count_feature_label = np.zeros((label_num, feature_nums, feature_values))
    count_label=[0]*label_num
    for i in range(x_train.shape[0]):
        label = int(y_train[i])
        count_label[label] += 1
        for j in range(x_train.shape[1]):
            feature_position = j
            feature_value = int(x_train[i,j])
            count_feature_label[label, feature_position, feature_value] += 1 
    prob_feature_label = np.zeros_like(count_feature_label)
    prior = np.zeros(label_num)
    for i in range(label_num):
        prob_feature_label[i,:,:] = count_feature_label[i,:,:] / count_label[i] 
        prior[i] = count_label[i] / sample_num 
    return prob_feature_label, prior

def model(data, prob_feature_label, prior):
    label_num = prob_feature_label.shape[0]; sample_num = data.shape[0]; feature_num = data.shape[1]
    prob = np.ones((label_num, sample_num))
    pred_label = np.zeros(sample_num)
    for i in range(label_num):
        for j in range(sample_num):
            for k in range(feature_num):
                idx_feature_value = int(data[j,k])
                if(prob_feature_label[i,k, idx_feature_value]<=0.001): 
                    prob_feature_label[i,k, idx_feature_value] = 0.001 
                prob[i, j] = prob[i,j] * prob_feature_label[i,k, idx_feature_value] 
            prob[i, j] = prob[i,j] * prior[i] 
    for i in range(sample_num):
        pred_label[i] = np.argmax(prob[:,i])
    return pred_label

def acc(pred_label, true_label): 
    acc = 1 - np.mean(np.abs(pred_label - true_label))
    return acc

def plot(var, title, color, ylabel):
    x = np.arange(0.1, 1.1, 0.1)
    plt.plot(x, var, label = 'time', color=color)
    plt.xlabel('Percentage of Training Data')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

def main():
    train = "../data/facedata/facedatatrain"
    train_label = "../data/facedata/facedatatrainlabels"
    test = "../data/facedata/facedatatest"
    test_label = "../data/facedata/facedatatestlabels"
    pool = 3
    x_train, y_train = process_data(train, train_label,pool)
    x_test, y_test = process_data(test, test_label,pool)
    amount = int(x_train.shape[0]/10)
    time_consume = []
    test_acc = []
    for i in range(10):
        start = time.time()
        prob_feature_label , prior = calc_prob(x_train[0:amount*(i+1)],y_train[0:amount*(i+1)],pool)
        end = time.time()
        pred_label = model(x_test, prob_feature_label, prior)
        accuracy = acc(pred_label, y_test)
        print("test accuracy:{}".format(accuracy))
        time_consume.append(end-start)
        test_acc.append(accuracy)
    plot(time_consume, title='FaceImage', color='blue', ylabel="Time(s)")
    plot(test_acc, title='FaceImage', color='red', ylabel='ACC')
main()


