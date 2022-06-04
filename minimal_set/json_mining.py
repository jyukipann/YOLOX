import json

if __name__ == "__main__":
    path = r"D:\Users\tanimoto.j\Downloads\via_project_4Jun2022_13h58m_coco (2).json"
    # path = r"D:\Users\tanimoto.j\Downloads\via_project_4Jun2022_13h58m_coco (4).json"
    # path = r"W:\tanimoto.j\dataset\flir\FLIR_ADAS_1_3\val\thermal_annotations.json"
    with open(path) as f:
        annotation = json.load(f)
    print(annotation.keys())
    print(len(annotation["annotations"]))
    # print(annotation["images"])
    # img_id = [1,2,14,146,217,226]
    for anno in annotation["annotations"]:
        x,y,w,h = anno['bbox']
        x1,y1,x2,y2 = x,y,x+w,y+h
        print(anno['image_id'], [x1,y1,x2,y2], w/h)