# Kaggle-Benetech 4th solution

## Hardware
Google Cloud Platform
* Debian 10.12
* a2-highgpu-1g (vCPU x 12, memory 85 GB)
* 1 x NVIDIA Tesla A100

## Data download
Download these datasets to ./input and unzip.
* [benetech-competition dataset](https://www.kaggle.com/competitions/benetech-making-graphs-accessible/data)
* [BARTLEY's dataset](https://www.kaggle.com/datasets/brendanartley/benetech-extra-generated-data)
* [our team's validation split](https://www.kaggle.com/datasets/fuumin621/benetech-split)
* [our team's generated graph dataset](https://www.kaggle.com/datasets/fuumin621/benetech-synthesis-4th)
  * if you want to make this dataset, please run all notebooks in [here](./synthesis) 
* [arial-font](https://www.kaggle.com/datasets/hammaadali/arial-font)
* [ICDAR 2022 train dataset](https://www.dropbox.com/s/85yfkigo5916xk1/ICPR2022_CHARTINFO_UB_PMC_TRAIN_v1.0.zip?dl=0)
* [ICDAR 2022 test dataset](https://www.dropbox.com/s/w0j0rxund06y04f/ICPR2022_CHARTINFO_UB_UNITEC_PMC_TEST_v2.1.zip?dl=0)


## Environment
```
docker-compose up -d --build
docker exec -it benetech bash
bash setup_yolox
```
## Preprocess
```
bash /work/preprocess/preprocess.sh
```

## Train 
```
bash /work/train/train.sh
```

## Submit
Upload all trained models below ./output to kaggle datasets

Here is our dataset for final sub.
https://www.kaggle.com/datasets/fuumin621/benetech-models-4th

Run this notebook.
https://www.kaggle.com/code/fuumin621/benetech-4th-submit/notebook



