import time
import os
import uuid
import datetime
import cv2 as cv
from running_average_detector import RunningAverageDetector
from inference import classify

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


fps = 60

# Show output feed
OUTPUT_STREAM = True

# number of images before the program exists
MAX_IMAGE_COUNT = 100

PRINT_LOGS = True

# to store images when motion is captured
CAPTURE_PATH = './motion_captures/'

CAMERA_SOURCE = "./test_videos/deer.mp4"

def main():
    """Starts the motion detection and classification loop. 
    
    It passes every frame from the camera feed to the 
    Motion Detector. If a frame does contain motion it
    is then passed to classification model to perform 
    inference.

    Raises:
        Exception: _description_
    """
    print("-"*10)
    print('Starting Motion Detection.')
    print("-"*10)
    
    
    cap = cv.VideoCapture(CAMERA_SOURCE)
    # cap = cv.VideoCapture(0)
    
    
    detector = RunningAverageDetector()
    previous_capture_time = 0
    previous_log_time = 0
    image_count = 0

    while True:
        ret, raw_image = cap.read()
        
        # Restart video if it has ended. 
        
        if not ret:
            # TODO:: In production `break` here. 
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            continue
        
        raw_image = cv.resize(raw_image, (1280, 720))
        motion, foreground_mask, frame = detector.apply(raw_image.copy())
        
        if OUTPUT_STREAM:
            cv.imshow('Difference', foreground_mask)
            cv.imshow('Original', frame)

            # Important
            if (cv.waitKey(30) == 27):
                break
        
        time_since_last_capture = time.time() - previous_capture_time
        time_since_last_log = time.time() - previous_log_time
        
        # Once an image is captured, the detectors waits for X seconds 
        # before considering new frames for detection.
        
        # Periodically log when not detecting for motion.
        if time_since_last_capture < DELAY_BETWEEN_CAPTURES and \
            time_since_last_log > 1:
                print(f'Motion detection stopped for {DELAY_BETWEEN_CAPTURES} seconds.')
                previous_log_time = time.time()
                
        
        if motion:
            
            # Prevent successive frames once motion is detected.
            if time_since_last_capture < DELAY_BETWEEN_CAPTURES:
                continue
            else:
                print(f'Motion detected at {datetime.datetime.now().strftime("%d%m%Y%H%M%S")}.')
            
            # Pass the cropped image to model.
            label, score = classify(raw_image)
            
            
            
            frame = cv.rectangle(frame, (0, 0), (250, 75), (0, 0, 0), -1)
            frame = cv.putText(frame, f'{label} {score:.2f}', (10, int(50)), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1, cv.LINE_AA)
            
            # Store image with classification. 
            if CAPTURE_MOTION_IMAGES:
                # save current frame
                dst_path = os.path.join(CAPTURE_PATH, f'{str(uuid.uuid1())}.jpg')
                # fg_dst_path = os.path.join(CAPTURE_PATH, f'{str(uuid.uuid1())}_foreground.jpg')                
                # cv.imwrite(fg_dst_path, foreground_mask)
                
                if not cv.imwrite(dst_path, frame):
                    raise Exception('Could not save image.')
                else:
                    # don't capture more than MAX_IMAGE_COUNT  
                    image_count += 1
                    if image_count > MAX_IMAGE_COUNT:
                        return
                    
                    # image captured, don't capture anymore images for the next X seconds
                    previous_capture_time = time.time()   

        
        if (cv.waitKey(30) == 27):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()