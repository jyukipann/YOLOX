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

def calc_iou(a, b):
    # aは1つの矩形を表すshape=(4,)のnumpy配列
    # array([xmin, ymin, xmax, ymax])
    # bは任意のN個の矩形を表すshape=(N, 4)のnumpy配列
    
    # 矩形aの面積a_areaを計算
    a_area = (a[2] - a[0] + 1) \
             * (a[3] - a[1] + 1)
    # bに含まれる矩形のそれぞれの面積b_areaを計算
    b_area = (b[2] - b[0] + 1) \
             * (b[3] - b[1] + 1)
    
    # aとbの矩形の共通部分(intersection)の面積を計算するために、
    # N個のbについて、aとの共通部分のxmin, ymin, xmax, ymaxを一気に計算
    abx_mn = max(a[0], b[0]) # xmin
    aby_mn = max(a[1], b[1]) # ymin
    abx_mx = min(a[2], b[2]) # xmax
    aby_mx = min(a[3], b[3]) # ymax
    # 共通部分の矩形の幅を計算。共通部分が無ければ0
    w = max(0, abx_mx - abx_mn + 1)
    # 共通部分の矩形の高さを計算。共通部分が無ければ0
    h = max(0, aby_mx - aby_mn + 1)
    # 共通部分の面積を計算。共通部分が無ければ0
    intersect = w*h
    
    # N個のbについて、aとのIoUを一気に計算
    iou = intersect / (a_area + b_area - intersect)

    return iou

def gen_img_id_index(ids):
    target_image_index = {}
    for i,img_id in enumerate(ids):
        try:
            target_image_index[int(img_id)].append(i)
        except:
            target_image_index[int(img_id)] = [i]
    return target_image_index

if __name__ == '__main__':
    print('compute')
    # ano = opencsv('flir_anotation_train_data.csv')
    # ano = opencsv('experiment_result/finetuned/flir_anotation_val_data.csv')
    ano = opencsv(r'experiment_result\before\flir_anotation_val_RGB_data.csv')
    header, ano = ano[0], ano[1:]
    # data = opencsv('flir_dataset_train_yolox_result.csv')
    # data = opencsv('experiment_result/finetuned/flir_dataset_val_thermal_yolox_result_finetuned.csv')
    data = opencsv(r'experiment_result\before\flir_dataset_val_RGB_yolox_result_sunny.csv')
    # data = opencsv(r'experiment_result\before\flir_dataset_val_thermal_yolox_result_sunny.csv')
    # data = opencsv('flir_dataset_val_thermal_yolox_result_finetuned.csv')
    _, data = data[0], data[1:]
    ano = np.array(ano)
    ano_paths = ano[:,0]
    # print(ano_paths[0])
    # exit(0)
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
    
    # out_path = 'experiment_result/finetuned/flir_dataset_val_thermal_yolox_val_iou.csv'
    # out_path = 'experiment_result/before/flir_dataset_val_thermal_val_iou_sunny.csv'
    out_path = r'experiment_result\before\flir_dataset_val_RGB_yolox_val_iou_sunny.csv'
    # out_path = 'yolox_val_iou_finetuned.csv'
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'image_id', 'iou_max'])

    debug = True
    debug = False
    # last_id = ano[-1,0]
    # anno_dic = {}
    # for i,row in enumerate(ano):
    #     image_id = int(row[0])
    #     try:
    #         anno_dic[image_id].append(i)
    #     except:
    #         anno_dic[image_id] = [i]
    # data_dic = {}
    # for i,row in enumerate(data):
    #     image_id = int(row[0])
    #     try:
    #         data_dic[image_id].append(i)
    #     except:
    #         data_dic[image_id] = [i]
    anno_dic = gen_img_id_index(ano[:,0])
    data_dic = gen_img_id_index(data[:,0])


    sunny_rgb_image_id = [156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,248,249,250,251,252,253,255,256,257,258,259,260,261,262,263,264,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,290,291,292,295,296,297,298,300,301,302,303,304,305,306,307,309,310,311,312,313,315,316,317,318,319,325,326,327,328,329,330,331,335,336,337,338,339,340,341,342,343,344,345,346,347,348,349,350,351,352,353,354,355,356,357,358,359,360,361,362,363,364,365,366,367,368,369,370,377,378,379,380,381,382,383,384,385,386,387,388,389,390,391,392,393,394,395,396,397,398,399,400,401,402,403,404,405,406,407,408,413,414,415,416,417,418,419,420,426,427,428,429,430,431,432,433,434,435,436,437,438,439,440,441,442,443,446,447,448,449,450,454,455,456,809,810,811,812,813,814,815,816,817,818,819,820,821,822,823,824,825,826,827,828,829,830,831,832,833,834,835,836,837,838,839,840,841,842,843,844,845,846,847,848,855,856,857,858,859,860,861,862,863,864,865,866,867,868,869,870,871,872,873,874,875,876,880,881,882,883,884,885,886,887,888,889,890,891,892,893,894,898,899,900,901,902,903,904,905,906,907,908,909,910,911,912,917,918,919,920,921,922,923,924,925,926,927,928,929,930,931,932,933,934,935,936,937,938,939,940,941,942,943,944,945,946,947,948,949,950,951,952,953,954,955,956,957,958,959,960,961,962,963,964,965,966,967,968,969,970,971,972,973,974,980,981,982,983,984,986,987,988,989,990,991,992,993,994,995,996,997,998,999,1000,1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012,1013,1014,1015,1016,1023,1024,1025,1026,1027,1028,1029,1030,1031,1032,1033,1034,1035,1036,1037,1038,1039,1040,1041,1042,1043,1044,1045,1046,1047,1048,1049,1050,1051,1052,1053,1054,1055,1056,1057,1058,1059,1060,1061,1062,1063,1064,1065,1066,1067,1068,1069,1070,1071,1072,1073,1074,1075,1076,1077,1078,1079,1080,1081,1084,1085,1086,1087,1088,1089,1090,1091,1092,1093,1094,1095,1096,1097,1098,1099,1100,1101,1102,1103,1104,1105,1106,1107,1108,1109,1110,1111,1113,1114,1115,1116,1117,1118,1119,1120,1121,1122,1123,1124,1125,1126,1127,1128,1129,1130,1133,1134,1135,1136,1137,1138,1139,1140,1141,1142,1143,1144,1145,1146,1147,1148,1149,1150,1151,1152,1153,1154,1155,1156,1157,1158,1159,1160,1161,1162,1163,1164,1165,1170,1171,1172,1173,1174,1175,1176,1177,1178,1179,1180,1181,1182,1183,1184,1185,1186,1187,1188,1189,1190,1191,1192,1193,1194,1196,1197,1198,1199,1200,1201,1203,1205,1207,1208,1209,1210,1211,1212,1213,1214,1215,1216,1217,1218,1219,1220,1221,1222,1223,1224,1225,1226,1227,1228,1229,1230,1231,1232,1233,1234,1235,1236,1237,1238,1239,1240,1241,1242,1243,1244,1245,1246,1248,1249,1250,1251,1252,1253,1254,1255,1256,1257,1258,1259,1260,1261,1262,1265,1266,1267,1268,1269,1270,1271,1272,1273,1274,1275,1276,1277,1278,1279,1280,1281,1282,1283,1284,1285,1286,1287,1288,1289,1290,1291,1292,1293,1294,1295,1296,1297,1298,1299,1300,1301,1302,1303,1304,1305,1306,1307,1308,1309,1310,1311,1312,1313,1314,1315,1316,1317,1318,1319,1323,1324,1325,1326,1327,1328,1329,1330,1331,1332,1333,1334,1335,1336,1337,1338,1339,1340,1341,1342,1343,1344,1345,1346,1347,1348,1349,1350,1351,1352,1353,1354,1355,1356,1357,1358,1360,1361,1362,1364,]

    img_ids = list(set(set(list(anno_dic.keys())) & set(list(data_dic.keys()))))
    for img_id in img_ids:
        print('\r',end='')
        ano_mask = anno_dic[img_id]
        ano_i = ano[ano_mask]
        if len(ano_i) < 1:
            continue
        ano_path_i = ano_paths[ano_mask[0]]
        print(ano_path_i)
        # exit(0)
        try:
            data_i = data[data_dic[img_id]]
        except:
            data_i = None
            print("data None",end="")
            continue
        if img_id not in sunny_rgb_image_id:
            continue
        # print(data_i)
        # exit(0  )
        # print(ano_i)
        # print(ano_path_i)
        # print(data_i)
        # data_i = data[data[:,0] == img_id]
        rows = []
        show_flag = False
        if img_id % 10 == 0:
            print(f"id:{img_id}",end='')
        for a in ano_i:
            debug_str = ""
            debug_str2 = ""
            if data_i is None:
                iou_max = 0
            else:
                # iou_max = float(np.max(iou_np(a[1:5], data_i[:, 1:5])))
                iou_max = 0
                
                for data_row in data_i:
                    dx1,dy1,dx2,dy2 = data_row[1:5].astype(int)
                    ax1,ay1,ax2,ay2 = a[1:5].astype(int)
                    iou = calc_iou([ax1,ay1,ax2,ay2],[dx1,dy1,dx2,dy2])
                    if iou_max < iou:
                        iou_max = iou

                    # if iou < 0.0001:
                    #     debug_str += f"{a[0]}"
                    #     debug_str += f"{iou}"
                    #     debug_str += f"{[ax1,ay1,ax2,ay2]}"
                    #     debug_str += f"{[dx1,dy1,dx2,dy2]}\n"
                    # else:
                    #     debug_str2 += f"{a[0]}"
                    #     debug_str2 += f"{iou}"
                    #     debug_str2 += f"{[ax1,ay1,ax2,ay2]}"
                    #     debug_str2 += f"{[dx1,dy1,dx2,dy2]}\n"

                        # print(a[0],iou,[ax1,ay1,ax2,ay2],[dx1,dy1,dx2,dy2])
                # if iou_max < 0.001:
                #     print(data_i.shape)
                #     print(debug_str)
                #     print(debug_str2)
        
            # print(a[1:5], data_i[:, 1:5])
            # print(iou_np(a[1:5], data_i[:, 1:5]))
            # print(type(np.max(iou_np(a[1:5], data_i[:, 1:5]))))
            # if img_id > 2:
            #     exit(0)
            # print(iou_max)
            if iou_max > 0.1:
                show_flag = True
            rows.append([ano_path_i,img_id,iou_max])
        # continue
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
        # exit(0)
    if debug:
        cv2.waitKey(0)

