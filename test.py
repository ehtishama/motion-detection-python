from model import classify
import cv2 
from PIL import Image
import numpy as np
import os
import time

# test_image = cv2.imread('./test_images/lion.jpg')
# test_image.load()
# np_test_image = np.asarray(test_image, dtype='int32')
# classify(test_image)
# print( np_test_image[0, 0, :] ) 


def classify_directory(path):
    """Classify all .jpeg images in a directory. 

    Args:
        path (string): path of the directory
    """    
    images = os.listdir(path)
    
    for image in images:
        image = cv2.imread(os.path.join(path, image))
        classify(image)
        cv2.imshow('image', image)
        cv2.waitKey(2*1000)
        
        # time.sleep(3)
        
    # print(images)
    
classify_directory("./motion_captures")

