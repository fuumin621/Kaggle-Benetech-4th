FROM gcr.io/kaggle-gpu-images/python:latest
RUN pip install omegaconf
RUN pip install timm
RUN pip install git+https://github.com/huggingface/transformers
RUN pip install polyleven
RUN pip uninstall torch torchvision torchaudio torchtext -y
RUN pip install torch torchvision