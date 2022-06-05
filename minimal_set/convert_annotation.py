from ctypes.wintypes import RGB
import numpy as np
import convert_rects
import csv

if __name__ == "__main__":

    # input path
    thermal_annotation_path = r'W:\tanimoto.j\workspace\GitHub\YOLOX\experiment_result\before\flir_anotation_val_data.csv'
    # output path
    RGB_annotation_path = r'W:\tanimoto.j\workspace\GitHub\YOLOX\experiment_result\before\flir_anotation_val_RGB_data.csv'

    thermal_annotation_data = convert_rects.opencsv(thermal_annotation_path)
    header, thermal_annotation_data = thermal_annotation_data[0], thermal_annotation_data[1:]
    thermal_annotation_data = np.array(thermal_annotation_data)
    thermal_annotation_image_paths = thermal_annotation_data[:,0]
    thermal_annotation_image_ids = thermal_annotation_data[:,1]
    thermal_annotation_rects = thermal_annotation_data[:,[2,3,4,5]].astype(int)
    thermal_annotation_conf = thermal_annotation_data[:,6].astype(float)
    thermal_annotation_category_ids = thermal_annotation_data[:,7].astype(int)

    # print(thermal_annotation_data[0,7])
    # exit(0)

    for i in range(thermal_annotation_data.shape[0]):
        x1,y1,x2,y2 = thermal_annotation_rects[i]
        x1,y1 = convert_rects.convert_point(x1,y1)
        x2,y2 = convert_rects.convert_point(x2,y2)
        thermal_annotation_rects[i] = np.array((x1,y1,x2,y2))
        thermal_annotation_image_paths[i] = thermal_annotation_image_paths[i].replace("thermal_8_bit","RGB")
        thermal_annotation_image_paths[i] = thermal_annotation_image_paths[i].replace("jpeg","jpg")


    # print(thermal_annotation_image_paths[0])
    # print(thermal_annotation_rects[0])

    with open(RGB_annotation_path,'w',newline="",encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(thermal_annotation_data.shape[0]):
            writer.writerow([
                thermal_annotation_image_paths[i],
                thermal_annotation_image_ids[i],
                *thermal_annotation_rects[i],
                thermal_annotation_conf[i],
                thermal_annotation_category_ids[i]
            ])
