import cv2
import numpy as np
import torch
from PIL import Image

def viz_image(image, target, tgt_size=(512, 512)):
    '''
    Args:
        image: Tensor (3, H, W)
        target: int (class label)
    Returns:
        None
    '''
    # Convert tensor to numpy
    image = image.permute(1, 2, 0).numpy().astype(np.uint8)
    image = cv2.resize(image, tgt_size)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
