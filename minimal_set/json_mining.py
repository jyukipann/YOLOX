import json

if __name__ == "__main__":
    
    path = r"W:\tanimoto.j\dataset\flir\FLIR_ADAS_1_3\val\RGB\RGB.json"
    with open(path) as f:
        annotation = json.load(f)
    # print(annotation.keys())
    # print(len(annotation["annotations"]))
    # print(annotation["images"])
    # img_ids = [1,2,14,146,217,226]
    rects = []
    for anno in annotation["annotations"]:
        x,y,w,h = anno['bbox']
        img_id = anno['image_id']
        file_name = annotation["images"][img_id-1]['file_name']
        x1,y1,x2,y2 = x,y,x+w,y+h
        rects.append([x1,y1,x2,y2])
        print(img_id, file_name, [x1,y1,x2,y2], w/h)
    for rect in rects:
        print(rect,",",sep="")

    path = r"W:\tanimoto.j\dataset\flir\FLIR_ADAS_1_3\val\thermal_8_bit\thermal.json"
    with open(path) as f:
        annotation = json.load(f)
    # print(annotation.keys())
    # print(len(annotation["annotations"]))
    # print(annotation["images"])
    # img_ids = [1,2,14,146,217,226]
    rects = []
    for anno in annotation["annotations"]:
        x,y,w,h = anno['bbox']
        img_id = anno['image_id']
        file_name = annotation["images"][img_id-1]['file_name']
        x1,y1,x2,y2 = x,y,x+w,y+h
        rects.append([x1,y1,x2,y2])
        print(img_id, file_name, [x1,y1,x2,y2], w/h)

    for rect in rects:
        print(rect,",",sep="")