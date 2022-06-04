import cv2
import numpy as np

RGB_image_points =\
    [
        [1589, 657, 1670, 831],
        [876, 713, 976, 792],
        [1317, 711, 1391, 771],
        [1085, 625, 1254, 953],
        [1363, 620, 1521, 959],
        [671, 711, 776, 795],
        [1454, 599, 1551, 875],
        [386, 669, 479, 897],
        [279, 674, 381, 909],
        [153, 687, 279, 908],
        [650, 669, 1008, 813],
        [1419, 720, 1768, 887],
        [148, 688, 506, 934],
        [1152, 724, 1298, 817],
        [349, 643, 1148, 1247],
        [239, 695, 573, 934],
        [1233, 539, 1403, 980],
    ]
print(len(RGB_image_points))

RGB_image_points = np.array(RGB_image_points,dtype=float)
print(RGB_image_points.shape)
RGB_image_points = RGB_image_points.reshape(-1, 2)

Thermal_img_points =\
    [
        [581, 204, 613, 283],
        [295, 226, 330, 253],
        [469, 224, 500, 253],
        [369, 191, 444, 320],
        [489, 191, 536, 326],
        [212, 225, 250, 254],
        [517, 186, 556, 293],
        [94, 210, 135, 304],
        [50, 210, 93, 306],
        [10, 209, 44, 304],
        [205, 205, 354, 267],
        [509, 229, 637, 300],
        [3, 213, 139, 309],
        [402, 229, 457, 266],
        [65, 198, 390, 439],
        [25, 220, 166, 314],
        [427, 158, 494, 340],
    ]
print(len(Thermal_img_points))
Thermal_img_points = np.array(Thermal_img_points,dtype=float)
print(Thermal_img_points.shape)
Thermal_img_points = Thermal_img_points.reshape(-1, 2)

# print('RGB_image_points = [')
# for p in RGB_image_points:
#     print(f'[{p[0]},{p[1]}],')
# print(']')

# print('Thermal_img_points = [')
# for p in Thermal_img_points:
#     print(f'[{p[0]},{p[1]}],')
# print(']')

print(RGB_image_points.shape)
print(Thermal_img_points.shape)
f_mat, mask = cv2.findFundamentalMat(RGB_image_points, Thermal_img_points, cv2.RANSAC)
retval, H1, H2 = cv2.stereoRectifyUncalibrated(RGB_image_points, Thermal_img_points,f_mat)
print(retval,H1,H2)

