import cv2
import numpy as np
import csv


# print('RGB_image_points = [')
# for p in RGB_image_points:
#     print(f'[{p[0]},{p[1]}],')
# print(']')

# print('Thermal_img_points = [')
# for p in Thermal_img_points:
#     print(f'[{p[0]},{p[1]}],')
# print(']')

# print(RGB_image_points.shape)
# print(Thermal_img_points.shape)

# RGB_image_size = (1800, 1600)
# half_rgb_size = (900, 800)
# Thermal_image_size = (640, 512)

M = [[ 2.34594784e+00 ,-1.06349344e-02,  1.59308517e+02],
 [-5.28646524e-02,  2.41858003e+00,  1.58510310e+02],
 [-3.03681653e-05, -1.43832960e-04,  1.00000000e+00]]
M = np.mat(M)
MINV = np.linalg.inv(M)

def convert_point(x,y):
    global M
    point = np.array([[[x,y]]],dtype=np.float32)
    converted_point = cv2.perspectiveTransform(point, M)
    # print(converted_point)
    x,y = converted_point.ravel().tolist()
    return int(x),int(y)

def opencsv(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data

def convert_thermal_image_to_RGB_size(thermal_img):
    global M
    return cv2.warpPerspective(thermal_img, M, (1800,1600))

def reverse_convert_RGB_image_to_RGB_size(RGB_img):
    global MINV
    return cv2.warpPerspective(RGB_img, MINV, (640, 512))


# M, mask = cv2.findHomography(Thermal_img_points, RGB_image_points, cv2.RANSAC, 5.0)
# print(M)

# thermal_img_path = r"W:\tanimoto.j\dataset\flir\FLIR_ADAS_1_3\val\thermal_8_bit\FLIR_08863.jpeg"
# rgb_img_path = r"W:\tanimoto.j\dataset\flir\FLIR_ADAS_1_3\val\RGB\FLIR_08863.jpg"

# thermal_img_path = r"W:\tanimoto.j\dataset\flir\FLIR_ADAS_1_3\val\thermal_8_bit\FLIR_10146.jpeg"
# rgb_img_path = r"W:\tanimoto.j\dataset\flir\FLIR_ADAS_1_3\val\RGB\FLIR_10146.jpg"

# thermal_img = cv2.imread(thermal_img_path)
# rgb_img = cv2.imread(rgb_img_path)
# rgb_img_convert = cv2.warpPerspective(thermal_img, M, RGB_image_size)
# blend_img = cv2.addWeighted(rgb_img, 0.5, rgb_img_convert, 0.5, 1)
# blend_img = cv2.resize(blend_img, half_rgb_size)
# cv2.imshow("blend_img", blend_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

if __name__ =='__main__':
    RGB_image_points =\
        [
            [1589, 657, 1670, 831],
            [876, 713, 976, 792],
            [1317, 711, 1391, 771],
            [1085, 625, 1254, 953],
            [1363, 620, 1521, 959],
            [671, 711, 776, 795],
            [270, 321, 393, 451],
            [223, 327, 346, 448],
            [304, 667, 440, 924],
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
            [186, 423, 270, 553],
            [334, 304, 349, 355],
            [1171, 191, 1207, 253],
            [270, 172, 284, 300],
        ]
    RGB_image_points = np.array(RGB_image_points, dtype=float)
    RGB_image_points = RGB_image_points.reshape(-1, 2)
    Thermal_img_points =\
        [
            [581, 204, 613, 283],
            [295, 226, 330, 253],
            [469, 224, 500, 253],
            [369, 191, 444, 320],
            [489, 191, 536, 326],
            [212, 225, 250, 254],
            [41, 65, 90, 119],
            [23, 62, 76, 118],
            [58, 203, 111, 315],
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
            [9, 104, 46, 159],
            [69, 57, 80, 80],
            [410, 17, 429, 44],
            [43, 2, 52, 59],
        ]
    Thermal_img_points = np.array(Thermal_img_points, dtype=float)
    Thermal_img_points = Thermal_img_points.reshape(-1, 2)
    M, mask = cv2.findHomography(Thermal_img_points, RGB_image_points, cv2.RANSAC, 5.0) 
    print(M)

exit(0)
if __name__ == '__main__':

    RGB_image_points =\
        [
            [1589, 657, 1670, 831],
            [876, 713, 976, 792],
            [1317, 711, 1391, 771],
            [1085, 625, 1254, 953],
            [1363, 620, 1521, 959],
            [671, 711, 776, 795],
            [270, 321, 393, 451],
            [223, 327, 346, 448],
            [304, 667, 440, 924],
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
            [186, 423, 270, 553],
            [334, 304, 349, 355],
            [1171, 191, 1207, 253],
            [270, 172, 284, 300],
            # [1505, 319, 1645, 472],
            # [1159, 351, 1361, 602],
            # [767, 680, 962, 858],
            # [388, 701, 630, 884],
        ]
    # print(len(RGB_image_points))

    RGB_image_points = np.array(RGB_image_points, dtype=float)
    # print(RGB_image_points.shape)
    RGB_image_points = RGB_image_points.reshape(-1, 2)

    Thermal_img_points =\
        [
            [581, 204, 613, 283],
            [295, 226, 330, 253],
            [469, 224, 500, 253],
            [369, 191, 444, 320],
            [489, 191, 536, 326],
            [212, 225, 250, 254],
            [41, 65, 90, 119],
            [23, 62, 76, 118],
            [58, 203, 111, 315],
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
            [9, 104, 46, 159],
            [69, 57, 80, 80],
            [410, 17, 429, 44],
            [43, 2, 52, 59],
            # [545, 71, 603, 133],
            # [402, 82, 486, 183],
            # [246, 215, 322, 286],
            # [91, 221, 189, 293],
        ]
    # print(len(Thermal_img_points))
    Thermal_img_points = np.array(Thermal_img_points, dtype=float)
    # print(Thermal_img_points.shape)
    Thermal_img_points = Thermal_img_points.reshape(-1, 2)

    point = Thermal_img_points[0]
    print(point)
    point = convert_point(*Thermal_img_points[0])
    print(point)
    print(RGB_image_points[0])

