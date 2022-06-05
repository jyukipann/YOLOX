import cv2
import numpy as np
import csv

def opencsv(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data

if __name__ == "__main__":
    data = opencsv(r'experiment_result\before\flir_dataset_val_RGB_yolox_result_sunny.csv')
    header,data = data[0],np.array(data[1:])
    image_id = data[:,1].astype(int)
    image_id = image_id.tolist()
    image_id = list(set(image_id))
    print(image_id)
    print('[', end="")
    for id in image_id:
        print(f'{id},',end="")
    print(']', end="")

"""

"""