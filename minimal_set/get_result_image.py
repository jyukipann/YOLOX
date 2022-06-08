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
    annotation_csv_path = "experiment_result/finetuned/flir_anotation_val_data.csv"
    annotation_csv_path = 'experiment_result/before/flir_anotation_val_RGB_data.csv'

    # target_result_csv_path = "experiment_result/finetuned/flir_dataset_val_thermal_yolox_result_finetuned.csv"
    # target_csv_path = "flir_dataset_val_thermal_yolox_result.csv"

    # thermal annotation
    # target_annotation_csv_path = "experiment_result/finetuned/flir_anotation_val_data.csv"
    # target_annotation_csv_path = "experiment_result/before/flir_anotation_val_data.csv"
    # target_result_csv_path = target_annotation_csv_path
    # target_result_csv_path = r"experiment_result\before\flir_dataset_val_thermal_yolox_result.csv"
    # target_result_csv_path = r"experiment_result\before\flir_dataset_val_thermal_yolox_result_sunny.csv"
    target_result_csv_path = r"experiment_result\before\flir_dataset_val_RGB_yolox_result_sunny.csv"
    # target_result_csv_path = None

    target_iou_csv_path = r"experiment_result\before\flir_dataset_val_RGB_yolox_val_iou_sunny.csv"
    # target_iou_csv_path = r"experiment_result\before\flir_dataset_val_thermal_val_iou_sunny.csv"
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
    img_ids = [157, 177, 179, 180, 181, 182, 183, 184, 185, 186, 188, 190, 191, 206, 207, 208, 209, 210, 211, 215, 219, 221, 222, 224, 226, 230, 232, 237, 238, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 266, 267, 268, 269, 307, 311, 312, 316, 317, 318, 319, 325, 326, 328, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 350, 351, 363, 364, 365, 378, 379, 380, 385, 386, 389, 390, 396, 397, 400, 401, 402, 403, 404, 405, 406, 407, 408, 413, 414, 415, 417, 418, 419, 420, 428, 429, 431, 432, 433, 435, 436, 437, 438, 441, 442, 443, 446, 447, 448, 449, 450, 454, 455, 456, 823, 824, 825, 829, 832, 833, 835, 836, 837, 838, 842, 846, 847, 848, 855, 856, 857, 858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 868, 871, 872, 874, 875, 876, 880, 881, 926, 932, 934, 937, 940, 946, 953, 959, 961, 962, 964, 967, 969, 980, 993, 994, 1008, 1009, 1010, 1025, 1026, 1030, 1031, 1032, 1035, 1046, 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1055, 1056, 1057, 1058, 1059, 1061, 1068, 1069, 1071, 1072, 1073, 1074, 1075, 1080, 1095, 1096, 1098, 1102, 1105, 1106, 1107, 1114, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1133, 1138, 1144, 1148, 1151, 1153, 1156, 1179, 1184, 1191, 1198, 1199, 1200, 1205, 1207, 1208, 1209, 1210, 1211, 1212, 1215, 1216, 1220, 1223, 1224, 1225, 1226, 1227, 1228, 1229, 1230, 1231, 1232, 1235, 1248, 1249, 1250, 1251, 1252, 1253, 1255, 1256, 1257, 1258, 1259, 1260, 1261, 1262, 1265, 1266, 1267, 1270, 1271, 1272, 1273, 1274, 1276, 1277, 1280, 1283, 1285, 1286, 1287, 1288, 1289, 1292, 1293, 1301, 1302, 1303, 1304, 1305, 1306, 1307, 1308, 1316, 1317, 1318, 1319, 1323, 1324, 1326, 1327, 1328, 1331, 1332, 1344, 1345, 1346, 1347, 1348, 1350, 1351, 1352, 1353, 1354, 1355, 1356, 1357, 1358, 1360, 1361, 1364]
    # img_ids = random.sample(img_ids,k=10)
    # img_ids = [832,157,1230,1302,1303,1355,1364,812,1319,191,190,1227]
    # img_ids = [img_ids[-1]]
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
        
        iou_min = 1
        anno_rect_mask = np.zeros(img.shape[:2],np.uint8)
        for anno_index_i,index in enumerate(anno_index):
            _,x1,y1,x2,y2,conf,cls_id = annotation_id_rects[index]
            iou = 0
            
            if target_iou_not_none: 
                try:
                    _,iou = target_id_iou[target_image_iou_index[img_id][anno_index_i]]
                    iou = float(iou)
                    if iou_min > iou:
                        iou_min = iou
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
            # cv2.imshow(window_titles[-1],img)
            if iou_min == 0:
                # cv2.imshow("has iou 0",img)
                # cv2.waitKey(50)
                title = window_titles[-1].replace(" ","_").replace(":","")
                cv2.imwrite(f"W:/tanimoto.j/workspace/GitHub/YOLOX/iou_0_results_rgb_day/{title}.jpg",img)
                # cv2.imwrite(f"W:/tanimoto.j/workspace/GitHub/YOLOX/iou_0_results_rgb_day/a.jpg",img)
                # print(f"W:/tanimoto.j/workspace/GitHub/YOLOX/iou_0_results_rgb_day/{title}.jpg")
                # exit(0)
    print(set(iou_0_image_ids))
    if len(window_titles) > 0:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        

    