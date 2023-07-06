import os
os.chdir("./../YOLOX")

voc_cls = '''
VOC_CLASSES = (
    "point",
)
'''
with open('./yolox/data/datasets/voc_classes.py', 'w') as f:
    f.write(voc_cls)

coco_cls = '''
COCO_CLASSES = (
    "point",
)
'''
with open('./yolox/data/datasets/coco_classes.py', 'w') as f:
    f.write(coco_cls)

os.system(f"cp -r ./../input/yolox_point ./")
os.system("python train.py -f ./../train/yolox_x_point.py -d 1 -b 16 --fp16 -c ./yolox_x.pth --cache")


