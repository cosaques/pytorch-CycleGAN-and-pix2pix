FROM python:3.10.6-buster

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y libgl1-mesa-glx

COPY models models
COPY data data
COPY checkpoints checkpoints
COPY options options
COPY server server
COPY util util

CMD python -m server.launch --dataroot "" --name "t-shirt-model-17-01-06-22" --model "test" --netG "unet_256" --dataset_mode "single" --norm "batch" --gpu_ids "-1" --host 0.0.0.0 --port $PORT
