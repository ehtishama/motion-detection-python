import cv2
import numpy as np

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

class NaiveDetector():
    def __init__(self) -> None:
        self.previous_frame = None
        self.ero_kernel_size = 3

    # returns (Boolean, Foreground, BbFrame)
    def apply(self,  frame):

        gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)
        
        if self.previous_frame is None:
            self.previous_frame = gray
            return (False, None, None)
        
        
        # find difference b/w current and prev frames 
        foreground = cv2.absdiff(gray, self.previous_frame)
        foreground_thresh = cv2.threshold(src=foreground, thresh=THRESHOLD_VALUE, maxval=255, type=cv2.THRESH_BINARY)[1]
        
        # apply morph-opening to remove small change
        ero_kernel = np.ones((self.ero_kernel_size, self.ero_kernel_size))
        foreground_thresh = cv2.morphologyEx(foreground_thresh, cv2.MORPH_OPEN, ero_kernel)

        # count no of pixels where motion with enough change 
        pixels_changed = np.sum(foreground_thresh == 255)
        lg_pixels_changed = np.log10(pixels_changed + 1)

        # update previous frame
        self.previous_frame = gray

        if lg_pixels_changed >= MIN_PIXELS_CHANGED:
            return (True, foreground_thresh, frame)
        else:
            return (False, None, None)