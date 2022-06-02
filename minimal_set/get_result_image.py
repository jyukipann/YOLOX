# python minimal_set/get_result_image.py
import csv
import numpy as np
import cv2

def opencsv(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data

def draw_rects(img, anno_rects, result_rects):
    anno_color = (255,0,0)
    result_color = (0,0,255)
    for rect in anno_rects:
        img = cv2.rectangle(img,
            pt1=rect[:2],
            pt2=rect[2:],
            color=anno_color,
            thickness=1,
        )
    for rect in result_rects:
        img = cv2.rectangle(img,
            pt1=rect[:2],
            pt2=rect[2:],
            color=result_color,
            thickness=1,
        )

def gen_img_id_index(ids):
    target_image_index = {}
    
    ids = 
    for i,img_id in enumerate(ids[:,0].tolist()):
        try:
            target_image_index[int(img_id)].append(i)
        except:
            target_image_index[int(img_id)] = [i]


if __name__ == "__main__":
    target_csv_path = "flir_dataset_val_thermal_yolox_result_finetuned.csv"
    # target_csv_path = "flir_dataset_val_thermal_yolox_result.csv"
    target_data = opencsv(target_csv_path)
    target_header = target_data[0]
    target_data = target_data[1:]
    target_data = np.array(target_data)
    target_data_img_paths_ids = target_data[:,:1]
    target_data_rects = target_data[:,1:6].astype(int)
    target_image_index = {}
    for i,img_id in enumerate(target_data_rects[:,0].tolist()):
        try:
            target_image_index[int(img_id)].append(i)
        except:
            target_image_index[int(img_id)] = [i]
    print(target_image_index)
    exit(0)

    anotation_csv_path = "flir_anotation_val_data.csv"
    anotation_data = opencsv(anotation_csv_path)
    anotation_data = np.array(anotation_data[1:])
    anotation_data_rects = anotation_data[:,1:6].astype(int)

    