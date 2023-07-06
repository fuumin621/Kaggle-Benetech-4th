apt install -y build-essential
apt install make
git clone https://github.com/Megvii-BaseDetection/YOLOX
cd YOLOX
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth
git clone https://github.com/cocodataset/cocoapi
pip install Cython
pip install pycocotools
pip3 install -U pip && pip3 install -r requirements.txt
pip3 install -v -e .
cp ./tools/train.py ./
apt-get install -y libgl1-mesa-dev
apt-get install -y libglib2.0-0