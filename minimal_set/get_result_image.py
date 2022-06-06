# python minimal_set/get_result_image.py
import csv
import numpy as np
import cv2
import flir_dataloader
import convert_rects

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
    for i,img_id in enumerate(ids):
        try:
            target_image_index[int(img_id)].append(i)
        except:
            target_image_index[int(img_id)] = [i]
    return target_image_index

if __name__ == "__main__":
    # annotation_csv_path = "experiment_result/finetuned/flir_anotation_val_data.csv"
    annotation_csv_path = 'experiment_result/before/flir_anotation_val_RGB_data.csv'

    # target_result_csv_path = "experiment_result/finetuned/flir_dataset_val_thermal_yolox_result_finetuned.csv"
    # target_csv_path = "flir_dataset_val_thermal_yolox_result.csv"

    # thermal annotation
    # target_annotation_csv_path = "experiment_result/finetuned/flir_anotation_val_data.csv"
    # target_annotation_csv_path = "experiment_result/before/flir_anotation_val_data.csv"
    # target_result_csv_path = target_annotation_csv_path
    # target_result_csv_path = r"experiment_result\before\flir_dataset_val_thermal_yolox_result.csv"
    target_result_csv_path = r"experiment_result\before\flir_dataset_val_RGB_yolox_result_sunny.csv"
    # target_result_csv_path = None

    target_iou_csv_path = r"experiment_result\before\flir_dataset_val_RGB_yolox_val_iou_sunny.csv"
    # target_iou_csv_path = None




    target_not_none = target_result_csv_path is not None
    target_iou_not_none = target_iou_csv_path is not None
    if target_not_none:
        target_data = opencsv(target_result_csv_path)
        target_data = np.array(target_data[1:])
        target_data_img_paths = target_data[:,0]
        target_id_rects = target_data[:,1:].astype(float)
        target_image_index = gen_img_id_index(target_id_rects[:,0].astype(int).tolist())

        if target_iou_not_none:
            target_data = opencsv(target_iou_csv_path)
            target_id_iou = np.array(target_data[1:])
            target_id_iou = target_id_iou[:,1:]
            target_image_iou_index = gen_img_id_index(target_id_iou[:,0].astype(int).tolist())
    annotation_data = opencsv(annotation_csv_path)
    annotation_data = np.array(annotation_data[1:])
    annotation_img_path = annotation_data[:,0]
    annotation_id_rects = annotation_data[:,1:].astype(float)
    annotation_image_index = gen_img_id_index(annotation_id_rects[:,0].astype(int).tolist())

    # iou == 0 ids
    img_ids = [1046, 1047, 1049, 1050, 1051, 1055, 1074, 1105, 1121, 1124, 1127, 1144, 1145, 1147, 1150, 1185, 163, 168, 174, 1199, 177, 1358, 181, 1209, 1210, 1211, 188, 190, 1225, 203, 204, 206, 207, 1232, 217, 218, 219, 220, 221, 222, 224, 1248, 1249, 1250, 1251, 1252, 230, 1253, 232, 1256, 235, 1259, 1260, 1276, 1277, 256, 257, 258, 260, 1285, 262, 1286, 264, 1287, 1289, 1292, 1301, 1302, 1303, 1310, 1311, 1316, 1317, 1318, 1319, 1326, 1327, 1328, 1329, 1333, 317, 318, 319, 830, 1344, 1345, 1346, 1347, 325, 1348, 1350, 1351, 1352, 1353, 331, 1354, 1355, 1356, 335, 1357, 337, 338, 339, 340, 341, 342, 343, 1361, 347, 875, 364, 876, 380, 385, 386, 901, 390, 396, 397, 400, 401, 402, 403, 404, 405, 406, 407, 413, 414, 415, 925, 417, 418, 419, 420, 936, 426, 428, 429, 431, 432, 433, 434, 435, 436, 437, 438, 441, 442, 443, 446, 447, 448, 449, 454, 455, 1005, 1008, 1015]
    import random
    img_ids = random.sample(img_ids,k=10)
    img_ids = [1319,1150]
    img_ids = list(set(set(list(annotation_image_index.keys())) & set(list(target_image_index.keys()))))
    img_ids = random.sample(img_ids,k=10)
    # if 1150 in img_ids:
    #     print(1150)
    # else:
    #     print("1150 does not exist")
    # exit(0)
    # img_ids = [163,168,174,177,181,188,190,203,204,206,207,217]
    # img_ids = target_image_index.keys() # sunny rgb image id  

    anno_color = (255,0,0)
    target_color = (0,0,255)
    window_titles = []
    # annotation_data_is_RGB = True
    annotation_data_is_RGB = False
    isImgShow = False
    isImgShow = True

    iou_0_image_ids = []
    for img_id in img_ids:
        try:
            anno_index = annotation_image_index[img_id]
        except:
            print(f"image id {img_id} annotation dose not exists, skip")
        img = cv2.imread(annotation_img_path[anno_index[0]])
        if img is None:
            print(f"image id {img_id} path dose not exists, skip")
            continue

        
        if annotation_data_is_RGB:
            thermal_img_path = annotation_img_path[anno_index[0]]
            thermal_img_path = thermal_img_path.replace("RGB","thermal_8_bit").replace("jpg","jpeg")
            thermal_img = cv2.imread(thermal_img_path)
            converted_thermal_img = convert_rects.convert_thermal_image_to_RGB_size(thermal_img)
            img = cv2.addWeighted(img,0.5,converted_thermal_img,0.5,1)
        
        anno_rect_mask = np.zeros(img.shape[:2],np.uint8)
        for anno_index_i,index in enumerate(anno_index):
            _,x1,y1,x2,y2,conf,cls_id = annotation_id_rects[index]
            iou = 0
            if target_iou_not_none: 
                try:
                    _,iou = target_id_iou[target_image_iou_index[img_id][anno_index_i]]
                except:
                    # print()
                    # print(f"{img_id} {anno_index_i}")
                    print(f"iamge id {img_id} result dose not exists, skip")
                    # print(f"{target_image_iou_index[img_id][anno_index_i]}")
                if float(iou) == 0:
                    # print()
                    w = x2 - x1
                    h = y2 - y1
                    print(f"image id {img_id} iou {iou} rect {x1,y1,x2,y2} area {w*h}")
                    iou_0_image_ids.append(img_id)
            x1,y1,x2,y2,cls_id = map(int, [x1,y1,x2,y2,cls_id])
            cv2.rectangle(anno_rect_mask,(x1,y1),(x2,y2),(1,),1)
            cv2.putText(
                anno_rect_mask,
                f"{float(iou):.3f}",
                org=(x1,y1),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(1,),
                thickness=1,
                lineType=cv2.LINE_4
            )
        target_rect_mask = np.zeros(img.shape[:2],np.uint8)
        if target_not_none:
            try:
                target_index = target_image_index[img_id]
            except:
                pass
            for target_id_rect in target_id_rects[target_index]:
                _, x1,y1,x2,y2,conf,cls_id = target_id_rect
                if annotation_data_is_RGB:
                    x1,y1 = convert_rects.convert_point(x1,y1)
                    x2,y2 = convert_rects.convert_point(x2,y2)
                x1,y1,x2,y2 = map(int, [x1,y1,x2,y2])
                cv2.rectangle(target_rect_mask,(x1,y1),(x2,y2),(1,),1)

        mask = anno_rect_mask + target_rect_mask
        anno_rect_img = np.zeros(img.shape,np.uint8)
        anno_rect_img[anno_rect_mask != 0] = anno_color
        target_rect_img = np.zeros(img.shape,np.uint8)
        target_rect_img[target_rect_mask != 0] = target_color
        rect_img = np.zeros(img.shape,np.uint8)
        rect_img[:,:,0] = mask
        rect_img[:,:,1] = mask
        rect_img[:,:,2] = mask
        rect_img[rect_img != 0] = (anno_rect_img[rect_img != 0] / rect_img[rect_img != 0]) + (target_rect_img[rect_img != 0] / rect_img[rect_img != 0])
        rect_img = rect_img.astype(np.uint8)
        # cv2.imshow("rect",rect_img)
        img[mask != 0,:] = rect_img[mask != 0,:]
        if not target_not_none:
            target_index = []
        window_titles.append(f"img_id:{img_id} anno:{len(anno_index)} result:{len(target_index)}")
        if img.shape == (1600,1800,3):
            img = cv2.resize(img,(900,800))
        if isImgShow:
            cv2.imshow(window_titles[-1],img)
    print(set(iou_0_image_ids))
    if len(window_titles) > 0:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        

    