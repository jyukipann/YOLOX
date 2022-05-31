# python minimal_set/calc_iou_csv.py

import csv
import numpy as np
import cv2

def opencsv(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data


# https://python-ai-learn.com/2021/02/08/ioufast/
# 矩形aと、複数の矩形bのIoUを計算
def iou_np(a, b):
    # aは1つの矩形を表すshape=(4,)のnumpy配列
    # array([xmin, ymin, xmax, ymax])
    # bは任意のN個の矩形を表すshape=(N, 4)のnumpy配列
    # 2次元目の4は、array([xmin, ymin, xmax, ymax])
    
    # 矩形aの面積a_areaを計算
    a_area = (a[2] - a[0] + 1) \
             * (a[3] - a[1] + 1)
    # bに含まれる矩形のそれぞれの面積b_areaを計算
    # shape=(N,)のnumpy配列。Nは矩形の数
    b_area = (b[:,2] - b[:,0] + 1) \
             * (b[:,3] - b[:,1] + 1)
    
    # aとbの矩形の共通部分(intersection)の面積を計算するために、
    # N個のbについて、aとの共通部分のxmin, ymin, xmax, ymaxを一気に計算
    abx_mn = np.maximum(a[0], b[:,0]) # xmin
    aby_mn = np.maximum(a[1], b[:,1]) # ymin
    abx_mx = np.minimum(a[2], b[:,2]) # xmax
    aby_mx = np.minimum(a[3], b[:,3]) # ymax
    # 共通部分の矩形の幅を計算。共通部分が無ければ0
    w = np.maximum(0, abx_mx - abx_mn + 1)
    # 共通部分の矩形の高さを計算。共通部分が無ければ0
    h = np.maximum(0, aby_mx - aby_mn + 1)
    # 共通部分の面積を計算。共通部分が無ければ0
    intersect = w*h
    
    # N個のbについて、aとのIoUを一気に計算
    iou = intersect / (a_area + b_area - intersect)
    return iou

if __name__ == '__main__':
    print('compute')
    # ano = opencsv('flir_anotation_train_data.csv')
    ano = opencsv('flir_anotation_val_data.csv')
    header, ano = ano[0], ano[1:]
    # data = opencsv('flir_dataset_train_yolox_result.csv')
    # data = opencsv('flir_dataset_val_thermal_yolox_result.csv')
    data = opencsv('flir_dataset_val_thermal_yolox_result_finetuned.csv')
    _, data = data[0], data[1:]
    ano = np.array(ano)
    ano_paths = ano[:,0]
    ano = ano[:,1:].astype(float)
    # x y w h -> x1 y1 x2 y2
    # ano[:, 3] += ano[:, 1]
    # ano[:, 4] += ano[:, 2]
    data = np.array(data)
    data = data[:,1:].astype(float)
    # x y w h -> x1 y1 x2 y2
    # data[:, 3] += data[:, 1]
    # data[:, 4] += data[:, 2]
    img_id = 0
    
    # out_path = 'yolox_val_iou.csv'
    out_path = 'yolox_val_iou_finetuned.csv'
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'image_id', 'iou_max'])

    debug = True
    while True:
        print('\r',end='')
        ano_mask = ano[:,0] == img_id
        ano_i = ano[ano_mask]
        if len(ano_i) < 1:
            break
        ano_path_i = ano_paths[ano_mask][0]
        data_i = data[data[:,0] == img_id]
        # print(ano_i)
        # print(ano_path_i)
        # print(data_i)
        rows = []
        show_flag = False
        for a in ano_i:
            iou_max = np.max(iou_np(a[1:5], data_i[:, 1:5]))
            # print(iou_max)
            if iou_max > 0.1:
                show_flag = True
            rows.append([ano_path_i,img_id,iou_max])
        if debug and show_flag:
            img = cv2.imread(ano_path_i)
            for a in ano_i:
                img = cv2.rectangle(img,
                        pt1=a[1:3].astype(int),
                        pt2=a[3:5].astype(int),
                        color=(0, 255, 0),
                        thickness=3,
                        lineType=cv2.LINE_4,
                        shift=0)
            for d in data_i:
                img = cv2.rectangle(img,
                        pt1=d[1:3].astype(int),
                        pt2=d[3:5].astype(int),
                        color=(255, 0, 0),
                        thickness=3,
                        lineType=cv2.LINE_4,
                        shift=0)
            cv2.imshow(ano_path_i, img)
            # cv2.waitKey(0)
        with open(out_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        img_id += 1
        # exit(0)
    if debug:
        cv2.waitKey(0)

