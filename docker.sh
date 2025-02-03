#!/bin/bash
IMAGE_NAME="mamltest"
IMAGE_TAG="latest"

# rocker --nvidia --x11 --network host   \
#  --volume .:/workspace  --volume /mnt/datasets:/workspace/datasets   \
#   --env DISPLAY=$DISPLAY  --env XAUTHORITY=/root/.Xauthority  \
#     --  $IMAGE_NAME:$IMAGE_TAG 
docker run -it --rm --gpus all --ipc=host --ulimit memlock=-1 --network="host"  \
   -e DISPLAY=$DISPLAY     -v /tmp/.X11-unix:/tmp/.X11-unix     -v $HOME/.Xauthority:/root/.Xauthority  \
    -v ./:/workspace     -v /mnt/:/workspace/dataset    $IMAGE_NAME:$IMAGE_TAG