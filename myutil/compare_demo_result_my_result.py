import csv
import enum
import numpy as np
import cv2
import matplotlib.pyplot as plt

def opencsv(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data


demo_result = r"experiment_result\demoOutput\rgb_result_demo.csv"
my_result = r"experiment_result\demoOutput\20220607_flir_dataset_val_RGB_yolox_result.csv"

demo_result_data = opencsv(demo_result)[1:]
my_result_data = opencsv(my_result)[1:]

demo_result_data = np.array(demo_result_data)
my_result_data = np.array(my_result_data)

# demo_result_data = demo_result_data[:,[0,1]]
# my_result_data = my_result_data[:,[0,1]]
file_names = list(set(demo_result_data[:,0]) | set(my_result_data[:,0]))
print(len(set(demo_result_data[:,0])))
print(len(set(my_result_data[:,0])))
print(len(file_names))

demo_filename_counts = dict.fromkeys(file_names, 0)
my_filename_counts = dict.fromkeys(file_names, 0)
# print(type(demo_result_data[0]))
# file_name,img_id,x1,y1,x2,y2,conf,cat = demo_result_data[0]
# exit(0)

for i,(file_name,img_id,x1,y1,x2,y2,conf,cat) in enumerate(demo_result_data):
    if demo_filename_counts[file_name] == 0:
        demo_filename_counts[file_name] = [i]
    else:
        demo_filename_counts[file_name].append(i)
for i,(file_name,img_id,x1,y1,x2,y2,conf,cat) in enumerate(my_result_data):
    if my_filename_counts[file_name] == 0:
        my_filename_counts[file_name] = [i]
    else:
        my_filename_counts[file_name].append(i)
# print(id(demo_filename_counts[list(demo_filename_counts.keys())[0]]))
# print(id(demo_filename_counts[list(demo_filename_counts.keys())[1]]))
# print(id(demo_filename_counts[file_name]))
# exit(0)
for file_name in file_names:
    if len(demo_filename_counts[file_name]) != len(my_filename_counts[file_name]):
        # print(file_name, len(demo_filename_counts[file_name]), len(my_filename_counts[file_name]))
        # exit(0)
        demo = demo_result_data[demo_filename_counts[file_name]]
        my =  my_result_data[my_filename_counts[file_name]]
        print(*demo)
        print(*my)
        # exit(0)