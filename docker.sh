$IMAGE_NAME="mamltest"
$IMAGE_TAG="latest"

rocker --nvidia --x11 --network host   \
 --volume .:/workspace  --volume /mnt/datasets:/workspace/datasets   \
  --env DISPLAY=$DISPLAY  --env XAUTHORITY=/root/.Xauthority  \
    --  $IMAGE_NAME:$IMAGE_TAG 