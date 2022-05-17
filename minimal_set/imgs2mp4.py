# python minimal_set/imgs2mp4.py
import cv2
import pathlib
import re

# out_path = pathlib.Path(r"W:\tanimoto.j\2021ROS\dataFromBag\2020-12-13-12-27-22\emergent_cam")
# out_path = pathlib.Path(r"W:\tanimoto.j\2021ROS\dataFromBag\2020-12-13-12-27-22\flir_l")
# img_path_root = pathlib.Path(r"W:\tanimoto.j\2021ROS\dataFromBag\2020-12-13-12-27-22\emergent_cam")
img_path_root = pathlib.Path(r"W:\tanimoto.j\2021ROS\_dataFromBag\2020-12-13-12-27-22\gs3")
img_path_list = list(img_path_root.glob('**/*.png'))
img_path_list = [p for p in img_path_root.glob('**/*') if re.search(r'.(png|jpg)', str(p))]
# print(img_path_list)
# exit(0)
test_img = cv2.imread(str(img_path_list[0]))
w,h,c = test_img.shape
print(w,h,c)
# exit(0)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# output file name, encoder, fps, size(fit to image size)
video = cv2.VideoWriter(str(img_path_root/'video.mp4'),fourcc, 30.0, (h, w))

if not video.isOpened():
    print("can't be opened")
    exit(0)

len_img = len(img_path_list)
for i, path in enumerate(img_path_list):
    # hoge0000.png, hoge0001.png,..., hoge0090.png
    img = cv2.imread(str(path))

    # can't read image, escape
    if img is None:
        print("can't read")
        break

    # add
    video.write(img)
    print(f'\r rendering video {int(((i+1)/len_img)*100)}%       ',end='')

video.release()
print('written')