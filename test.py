from inference import classify
import cv2 
from PIL import Image
import numpy as np

test_image = cv2.imread('./test_images/lion.jpg')
# test_image.load()

# np_test_image = np.asarray(test_image, dtype='int32')

classify(test_image)

# print( np_test_image[0, 0, :] ) 


