import cv2 as cv
import numpy as np
import time
import os
import uuid
import datetime
from utils import clear_console

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

    previous_capture_time = 0
    previous_frame = None
    image_count = 0

    while True:

        motion = False
        ret, raw_image = cap.read()
        
        # loop back to start of the video
        if not ret:
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            continue
        
        raw_image = cv.resize(raw_image, (512, 512))
        current_frame = cv.cvtColor(src=raw_image, code=cv.COLOR_BGR2GRAY)
        
        if previous_frame is None:
            previous_frame = current_frame
            continue
        
        # find difference b/w current and prev frames 
        diff_frame = cv.absdiff(current_frame, previous_frame)
        diff_frame_processed = cv.threshold(src=diff_frame, thresh=THRESHOLD_VALUE, maxval=255, type=cv.THRESH_BINARY)[1]
        
        # apply morph-opening to remove small change
        ero_kernel = np.ones((ERO_KERNEL_SIZE, ERO_KERNEL_SIZE))
        diff_frame_processed = cv.morphologyEx(diff_frame_processed, cv.MORPH_OPEN, ero_kernel)
        
        
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
                       
        # detect and segment shapes
        # diff_frame_contours, hierarchy = cv.findContours(image=diff_frame_processed, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)
        

        # for contour in diff_frame_contours:
        #     if cv.contourArea(contour) < MIN_CONTOUR_AREA:
        #         continue
            
        #     (x, y, w, h) = cv.boundingRect(contour)

        #     bounding_box_offset = 0

            
        #     cv.rectangle(img=raw_image, pt1=(x, y), pt2=(x+w+bounding_box_offset, y+h+bounding_box_offset), color=(0, 255, 0), thickness=1)
            
            
        previous_frame = current_frame
        
        if OUTPUT_STREAM:
            cv.imshow('Video', raw_image) # press escape to exit
            cv.imshow('Difference', diff_frame_processed)

        if (cv.waitKey(30) == 27):
            break

    cap.release()
    cv.destroyAllWindows()
 
