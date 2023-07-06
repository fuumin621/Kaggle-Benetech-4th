import os
os.chdir("./../YOLOX")

voc_cls = '''
VOC_CLASSES = (
  "plot-bb",
  "chart_title",
  "axis_title",
  "x_label",
  "y_label",
  "other",
)
'''
with open('./yolox/data/datasets/voc_classes.py', 'w') as f:
    f.write(voc_cls)

coco_cls = '''
COCO_CLASSES = (
  "plot-bb",
  "chart_title",
  "axis_title",
  "x_label",
  "y_label",
  "other",
)
'''
with open('./yolox/data/datasets/coco_classes.py', 'w') as f:
    f.write(coco_cls)

os.system(f"cp -r ./../input/yolox_plotbb ./")
os.system("python train.py -f ./../train/yolox_x_plotbb.py -d 1 -b 16 --fp16 -c ./yolox_x.pth --cache")
