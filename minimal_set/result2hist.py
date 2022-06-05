# python minimal_set/result2hist.py
import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt

def opencsv(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data


# draw histgram

if __name__ == "__main__":
    # target_csv_path = "flir_dataset_val_thermal_yolox_result.csv"
    # target_csv_path = "flir_dataset_val_thermal_yolox_result_finetuned.csv"
    # target_csv_path = "experiment_result/before/yolox_val_iou.csv"
    # target_csv_path = r"experiment_result\before\flir_dataset_val_RGB_yolox_val_iou.csv"
    # target_csv_path = r"experiment_result\before\flir_dataset_val_RGB_yolox_val_iou_sunny.csv"
    target_csv_path = r"experiment_result\before\flir_dataset_val_thermal_val_iou_sunny.csv"
    # target_csv_path = "experiment_result/finetuned/flir_dataset_val_thermal_yolox_val_iou.csv"
    target_data = opencsv(target_csv_path)
    target_header = target_data[0]
    target_data = target_data[1:]
    target_data = np.array(target_data)
    target_data_img_paths_ids = target_data[:,:2]
    # target_data_rects = target_data[:,2:6].astype(int)
    target_data_iou = target_data[:,2].astype(float)
    print(f"ious {target_data_iou.shape}")


    # for row in target_data_iou:
    #     if row < 0.3:
    #         print(row)

    rng = [i/10 for i in range(0,10)]
    all_count = target_data_iou.shape[0]
    for i in rng:
        print(f"{i}:{target_data_iou[(target_data_iou >= i) & (target_data_iou < i+0.1)].shape[0]}")
    plt.hist(target_data_iou,bins=100, label="thermal 8bit iou")
    # plt.title('遠赤外線8bit画像のyoloxの推定結果のIOUのヒストグラム', fontname="MS Gothic")
    plt.title('遠赤外線画像のyoloxの推定結果のIOUのヒストグラム', fontname="MS Gothic")
    plt.xlabel("IOU", fontname="MS Gothic")
    plt.ylabel("度数", fontname="MS Gothic")
    plt.legend()
    plt.show()
    exit(0)

    target_csv_path = "flir_dataset_val_thermal_yolox_result_finetuned.csv"
    target_csv_path = "flir_dataset_val_thermal_yolox_result.csv"
    target_data = opencsv(target_csv_path)
    target_header = target_data[0]
    target_data = target_data[1:]
    target_data = np.array(target_data)
    target_data_img_paths_ids = target_data[:,:2]
    target_data_rects = target_data[:,2:6].astype(int)

    anotation_csv_path = "flir_anotation_val_data.csv"
    anotation_data = opencsv(anotation_csv_path)
    anotation_data = np.array(anotation_data[1:])
    anotation_data_rects = anotation_data[:,2:6].astype(int)

    # print(target_data_img_paths_ids[0])
    img_ids = [0]
    windows = []
    for id in [1364]:
        result_index = target_data_img_paths_ids[:,1] == f"{id}"
        ano_index = anotation_data[:,1] == f"{id}"
        path = target_data_img_paths_ids[result_index,:][0,0]
        # print(path)
        img = cv2.imread(path)
        if img is None:
            continue
        result_rects = target_data_rects[result_index, :]
        ano_rects = anotation_data_rects[ano_index, :]
        # print(rect)
        for rect in ano_rects:
            img = cv2.rectangle(img,
                pt1=rect[:2],
                pt2=rect[2:],
                color=(255, 0, 0),
                thickness=3,
            )
        for rect in result_rects:
            img = cv2.rectangle(img,
                pt1=rect[:2],
                pt2=rect[2:],
                color=(0, 0, 255),
                thickness=2,
            )
        cv2.imshow(f"id{id}",img)
        windows.append(f"id{id}")
    if len(windows) > 0:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    