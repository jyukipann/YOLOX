import cv2
import numpy as np

id1227rect = [
    [1175,674,1342,809],
    [1589,697,1799,877],
    [959,680,1059,777],
    [1048,642,1143,831],
    [116,604,194,803],
    [883,709,937,753],
    [61,616,122,797],
    [1118,709,1183,789],
    [658,690,746,782],
    [740,701,781,766],
    [783,711,816,753],
    [644,702,675,791],
    [934,713,965,751],
    [801,713,831,747],
    [834,710,863,742],
    [816,711,846,743],
    [851,711,878,739],
]

if __name__ == "__main__":
    img = cv2.imread(r"\\aka\work\tanimoto.j\dataset\flir\FLIR_ADAS_1_3\val\RGB\FLIR_10090.jpg")
    equ = np.zeros(img.shape,np.uint8)
    for c in range(3):
        equ[:,:,c] = cv2.equalizeHist(img[:,:,c])
    for rect in id1227rect:
        x1,y1,x2,y2 = rect
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),1)
    res = np.hstack((img,equ)) #stacking images side-by-side
    cv2.imshow("img",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imwrite('res.png',res)
    # cv2.imwrite('eqHistId1227.png',equ)