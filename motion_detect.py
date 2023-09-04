import cv2 as cv
import numpy as np
import time
import os
import uuid
import datetime
from utils import clear_console
from naive_detector import NaiveDetector

# Config

# determines the threshold value to binarize
# the difference frame
THRESHOLD_VALUE = 20

# kernel size for perfoming erosion 
# on the differnece frame to remove small noises
ERO_KERNEL_SIZE = 2

# value between [0, 1]
# determies the sensitivity of detection
# 1 means the system will detect slightest motion
# 0 means even the won't be detected
MOTION_SENSITIVITY = 0.6

# determines the minimun numbers of pixels that need to change
# on the log 10 scale to be considered as motion
# 10 means at least 1e^10 pixels need to change for motion to be recorded
# 1 means at leaset 10 pixels need to change for motion to be recorded
MIN_PIXELS_CHANGED = ( 1 -  MOTION_SENSITIVITY) * 8  

# the minimum area for the contours 
# of the difference frame to be considered
MIN_CONTOUR_AREA = 400

# to avoid continous frames being captures 
DELAY_BETWEEN_CAPTURES = 5

# whether to save images with motion or not
CAPTURE_MOTION_IMAGES =  True

# to store images when motion is captured
CAPTURE_PATH = './motion_captures/'

fps = 60

# Show output feed
OUTPUT_STREAM = True

# number of images before the program exists
MAX_IMAGE_COUNT = 100

PRINT_LOGS = True

def detect_motion(callback):
    print('Starting motion detection...')
    
    # get live feed from webcam
    # cap = cv.VideoCapture("./test_videos/trap-camera-video1.mp4")
    cap = cv.VideoCapture(0)
    detector = NaiveDetector()
    previous_capture_time = 0
    image_count = 0

    while True:
        ret, raw_image = cap.read()
        
        # loop back to start of the video
        if not ret:
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            continue
        
        raw_image = cv.resize(raw_image, (512, 512))
        motion, foreground_mask, frame = detector.apply(raw_image)
        
        if motion:
            print('motion')
            cv.imshow('Difference', foreground_mask)
            cv.imshow('Original', frame)
        else:
            cv.imshow('Original', raw_image)
        
        if (cv.waitKey(30) == 27):
            break

        continue
        
        # this prevents from continually capturing
        # each frame once motion is detected
        time_since_last_capture = time.time() - previous_capture_time
        if time_since_last_capture <= DELAY_BETWEEN_CAPTURES:
            
            # display output
            if OUTPUT_STREAM:
                cv.imshow('Video', raw_image) # press escape to exit
                cv.imshow('Difference', diff_frame_processed)
            
            if (cv.waitKey(30) == 27):
                break
            continue
        
        # count no of pixels where motion with enough change 
        pixels_changed = np.sum(diff_frame_processed == 255)
        lg_pixels_changed = np.log10(pixels_changed + 1)
        
        if PRINT_LOGS and lg_pixels_changed > 1:
            # clear_console()
            print(f'{datetime.datetime.now().strftime("%d%m%Y%H%M%S")} ', end='')
            print(f'{"Motion detected" if lg_pixels_changed > MIN_PIXELS_CHANGED else "No motion detected"}. ', end='')
            print(f'1e^{lg_pixels_changed:.1f} pixels changed. \n')

        if lg_pixels_changed >= MIN_PIXELS_CHANGED:            
            motion = True

            if CAPTURE_MOTION_IMAGES:
                # save current frame
                dst_path = os.path.join(CAPTURE_PATH, f'{str(uuid.uuid1())}.jpg')
                
                if not cv.imwrite(dst_path, raw_image):
                    raise Exception('Could not write image')
                else:
                    # don't capture more than MAX_IMAGE_COUNT  
                    image_count += 1
                    if image_count > MAX_IMAGE_COUNT:
                        return
                    
                    # image captured, don't capture anymore images for the next X seconds
                    previous_capture_time = time.time()
        
            callback(raw_image)
                                 
        previous_frame = current_frame
        
        if OUTPUT_STREAM:
            cv.imshow('Video', raw_image) # press escape to exit
            cv.imshow('Difference', diff_frame_processed)

        if (cv.waitKey(30) == 27):
            break

    cap.release()
    cv.destroyAllWindows()
 
def dummy_func(image):
    pass

if __name__ == '__main__':
    detect_motion(dummy_func)