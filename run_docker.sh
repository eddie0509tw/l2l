WORKSPACE_DIR="./"
IMAGE_NAME=mamltest
IMAGE_VERSION=latest

docker run -it --rm --gpus all --ipc=host --ulimit memlock=-1 \
    --network="host" \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $HOME/.Xauthority:/root/.Xauthority \
    -v ./:/workspace \
    -v /mnt/datasets:/workspace/datasets \
    $IMAGE_NAME:$IMAGE_VERSION \
    jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --NotebookApp.token=''
    

# rocker --nvidia --x11 --network host     --volume .:/workspace     --volume /mnt/datasets:/workspace/datasets    --env DISPLAY=$DISPLAY     --env XAUTHORITY=/root/.Xauthority     -- mamltest:latest


#### For kernel inside the docker container to be visible in jupyter notebook:
# 1. Select the "select kernel" option in the jupyter notebook
# 2. Select the existing kernel
# 3. Type http://localhost:8888
