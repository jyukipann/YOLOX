import csv
from unicodedata import category
import flir_dataloader


if __name__ == '__main__':
    print('output')
    # dataset = flir_dataloader.FlirDataset(dataset_dir="train")
    # output_file_name = 'flir_anotation_train_data.csv'
    # dataset = flir_dataloader.FlirDataset(dataset_dir="val")
    # output_file_name = 'flir_anotation_val_data.csv'
    with open(output_file_name, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'image_id', 'x1', 'y1', 'x2', 'y2', 'conf', 'class_id'])
    for i in range(len(dataset)):
        img, target, img_info, img_id = dataset[i]
        rows = []
        for t in target:
            t['bbox'][2] += t['bbox'][0]
            t['bbox'][3] += t['bbox'][1]
            category_id = t['category_id']
            row = [img_info['file_name'], i] + t['bbox'] + [0] + [category_id]
            rows.append(row)
        with open(output_file_name, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
