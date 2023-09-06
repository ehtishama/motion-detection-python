import cv2
import numpy as np
  
alpha = 0.3
min_widht = 100
max_width = 400
min_height = 100
max_height = 400

class RunningAverageDetector():

    def __init__(self) -> None:
        self.alpha = alpha
        self.running_avg = None
        self.thresh = 10
        self.min_width = min_widht
        self.min_height = min_height
        self.max_width = max_width
        self.max_height = max_height


    # compares frame to running avg
    # return true if there's substantial difference
    # (boolean, diff_mask, raw_img)
    def apply(self, frame):

        # 
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.GaussianBlur(gray_img, (21, 21), 0)

        # initialize background if not already
        if self.running_avg is None:
            self.running_avg = gray_img.copy().astype('float')
            return (False, gray_img, frame)
        
        cv2.accumulateWeighted(gray_img, self.running_avg, alpha)
      
        background = cv2.convertScaleAbs(self.running_avg)

        foreground = cv2.absdiff(gray_img, background)
        foreground_thresh = cv2.threshold(src=foreground, thresh=self.thresh, maxval=255, type=cv2.THRESH_BINARY)[1]
        foreground_thresh = cv2.dilate(foreground_thresh, None, iterations=2)

        contours, _ = cv2.findContours(foreground_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return (False, foreground_thresh, frame)
    
        contour_areas = [cv2.contourArea(c) for c in contours]
        maxIndex = np.argmax(contour_areas)
        largest_contour = contours[maxIndex]
    

        (x, y, w, h) = cv2.boundingRect(largest_contour)

        bounding_box_offset = 0
        cv2.rectangle(img=frame, pt1=(x, y), pt2=(x+w+bounding_box_offset, y+h+bounding_box_offset), color=(0, 255, 0), thickness=1)

        # print(f'w: {w}, h:{h}')
        
        if w < self.min_width or h < self.min_height:
            return (False, foreground_thresh, frame)
        

        
        return (True, foreground_thresh, frame)
    

def run_detector():
    # capture frames from a camera
    cap = cv2.VideoCapture(0)
    detector = RunningAverageDetector()
    
    
    # loop runs if capturing has been initialized. 
    while(1):
        # reads frames from a camera 
        _, img = cap.read()
        
        motion, foreground, frame = detector.apply(img)

        if motion: 
            cv2.imshow('InputWindow', frame)
            cv2.imshow('DiffWindow', foreground)
        else: 
            cv2.imshow('InputWindow', img)
            # cv2.imshow('DiffWindow', None)
                
        
        # Wait for Esc key to stop the program 
        k = cv2.waitKey(30) & 0xff
        if k == 27: 
            break
    
    # Close the window 
    cap.release() 
        
    # De-allocate any associated memory usage 
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run_detector()