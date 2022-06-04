# python minimal_set/get_result_image.py
import csv
import numpy as np
import cv2
import flir_dataloader

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

    target_result_csv_path = "experiment_result/finetuned/flir_dataset_val_thermal_yolox_result_finetuned.csv"
    # target_csv_path = "flir_dataset_val_thermal_yolox_result.csv"
    
    target_iou_csv_path = "experiment_result/finetuned/flir_dataset_val_thermal_yolox_val_iou.csv"

    target_data = opencsv(target_result_csv_path)
    target_data = np.array(target_data[1:])
    target_data_img_paths = target_data[:,0]
    target_id_rects = target_data[:,1:].astype(float)
    target_image_index = gen_img_id_index(target_id_rects[:,0].astype(int).tolist())

    target_data = opencsv(target_iou_csv_path)
    target_id_iou = np.array(target_data[1:])
    target_id_iou = target_id_iou[:,1:]

    annotation_data = opencsv(annotation_csv_path)
    annotation_data = np.array(annotation_data[1:])
    annotation_img_path = annotation_data[:,0]
    annotation_id_rects = annotation_data[:,1:].astype(float)
    annotation_image_index = gen_img_id_index(annotation_id_rects[:,0].astype(int).tolist())

    img_ids = [0]
    anno_color = (255,0,0)
    target_color = (0,0,255)
    window_titles = []
    for img_id in img_ids:
        anno_index = annotation_image_index[img_id]
        img = cv2.imread(annotation_img_path[anno_index[0]])
        anno_rect_mask = np.zeros(img.shape[:2],np.uint8)
        for index in anno_index:
            _,x1,y1,x2,y2,conf,cls_id = annotation_id_rects[index]
            _,iou = target_id_iou[index]
            x1,y1,x2,y2,cls_id = map(int, [x1,y1,x2,y2,cls_id])
            cv2.rectangle(anno_rect_mask,(x1,y1),(x2,y2),(1,),1)
            cv2.putText(
                anno_rect_mask,
                f"{float(iou):.2f}",
                org=(x1,y1),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(1,),
                thickness=1,
                lineType=cv2.LINE_4
            )

        target_index = target_image_index[img_id]
        target_rect_mask = np.zeros(img.shape[:2],np.uint8)
        for target_id_rect in target_id_rects[target_index]:
            _, x1,y1,x2,y2,conf,cls_id = target_id_rect
            x1,y1,x2,y2 = map(int, [x1,y1,x2,y2])
            cv2.rectangle(target_rect_mask,(x1,y1),(x2,y2),(1,),1)

        mask = anno_rect_mask + target_rect_mask
        anno_rect_img = np.zeros(img.shape,np.uint8)
        target_rect_img = np.zeros(img.shape,np.uint8)
        anno_rect_img[anno_rect_mask != 0] = anno_color
        target_rect_img[target_rect_mask != 0] = target_color
        rect_img = np.zeros(img.shape,np.uint8)
        rect_img[:,:,0] = mask
        rect_img[:,:,1] = mask
        rect_img[:,:,2] = mask
        rect_img[rect_img != 0] = (anno_rect_img[rect_img != 0] / rect_img[rect_img != 0]) + (target_rect_img[rect_img != 0] / rect_img[rect_img != 0])
        rect_img = rect_img.astype(np.uint8)
        cv2.imshow("rect",rect_img)
        img[mask != 0,:] = rect_img[mask != 0,:]
        window_titles.append(f"img_id:{img_id} anno:{len(anno_index)} result:{len(target_index)}")
        cv2.imshow(window_titles[-1],img)
    if len(window_titles) > 0:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        

    