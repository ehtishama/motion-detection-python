import cv2 as cv

cap = cv.VideoCapture(0)

while True:
    ret, image = cap.read()
    
    if not ret:
        print('Failed to get frame from camera. Exiting.')
        break
    
    cv.imshow('Live Camera', image)
    if cv.waitKey(30) == 27:
        break
    

cap.release()
cv.destroyAllWindows()
        
    
    