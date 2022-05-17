import pathlib
import json
import torch
import torchvision
import pycocotools
from pycocotools.coco import COCO
from torch.utils.data import Dataset
import cv2
import PIL

# params
dataset_root = pathlib.Path(r"\\aka\work\tanimoto.j\dataset\flir\FLIR_ADAS_1_3")
dataset_dir = "val"

anotation_path = list((dataset_root/dataset_dir).glob('*.json'))[0]
# print('anotation_path :',anotation_path)
anotation_file = open(anotation_path, 'r')
anotation = json.load(anotation_file)

# print(anotation.keys())
# print(anotation['annotations'][0:7])
# print(anotation['annotations'][1])
# print(len(anotation['annotations']))

# print(anotation['images'][0])
# print(anotation['images'][1])
# print(len(anotation['images']))

class FlirDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_root=r"\\aka\work\tanimoto.j\dataset\flir\FLIR_ADAS_1_3", dataset_dir="val", transform=None, RGB=False):
        self.RGB = RGB
        self.dataset_root = pathlib.Path(dataset_root)
        self.dataset_dir = dataset_dir
        anotation_path = list((self.dataset_root/self.dataset_dir).glob('*.json'))[0]
        # anotation_file = open(anotation_path, 'r')
        # anotation = json.load(anotation_file)
        self.coco = COCO(str(anotation_path))
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        # id_ = self.ids[index]
        img_id = self.coco.getImgIds(index)
        img_info = self.coco.loadImgs(index)[0]
        ano_id = self.coco.getAnnIds(index)
        target = self.coco.loadAnns(ano_id)
        # print(img_id, ano_id)
        # print()
        # self.coco.showAnns(target)
        img_path = str(self.dataset_root/self.dataset_dir/img_info['file_name'])
        if self.RGB:
            img_path = img_path.replace('thermal_8_bit','RGB')
            img_path = img_path.replace('.jpeg','.jpg')
            # print(img_path)
            img = cv2.imread(img_path)
            if img is None:
                return None, None, None, None
            img_info['height'] = img.shape[0]
            img_info['width'] = img.shape[1]
        else:
            img = cv2.imread(img_path)
        # img = PIL.Image.open(str(self.dataset_root/self.dataset_dir/img_info['file_name']))
        img_info['raw_img'] = img
        # img = torchvision.transforms.functional.to_tensor(img)
        img_info['file_name'] = img_path
        # img_info["ratio"] = ratio
        if self.transform is not None:
            img, target = self.transform(img, target)
        return img, target, img_info, img_id

if __name__ == '__main__':
    fir_dataset = FlirDataset(RGB=True)
    img, target, img_info, img_id = fir_dataset[0]
    # print(fir_dataset[0])
    # ids = [a[] anotation['annotations']]
    # img.show()
    # print(target)
    print(img_info)
    # print(img_id)