import numpy as np


def load_label(label_file):
    f = open(label_file)
    line = f.readlines()
    line = [int(item.strip()) for item in line]
    sample_num = len(line)
    return line, sample_num

def load_sample(sample_file, sample_num):
    f = open(sample_file)
    line = f.readlines()
    file_length = int(len(line))  #total sample amount 
    width = int(len(line[0]))  #the row length of one single image
    length = int(file_length/sample_num)  #the column length of one single image
    all_image = []
    print(len(line[0]),file_length/sample_num )
    print(width, length)
    for i in range(sample_num):
        single_image = np.zeros((length,width))
        count=0
        for j in range(length*i,length*(i+1)):  #length = 70
            single_line=line[j]
            #print(len(single_line))
            for k in range(len(single_line)):
                if(single_line[k] == "+" or single_line[k] == "#"):
                    single_image[count, k] = 1  #transform image data into binary format （0 for empty pixels and 1 for black and grey pixels) 
            count+=1        
        all_image.append(single_image)  #dimension of all_image_data : [total_sample_mount, rows, columns]
    #print(all_image_data[0])  #image pixels consist of '+' , '#' and empty signs, which mean 'black', 'grey' and white pixel respectively I guess
    return all_image

