import cv2
import pickle

cap = cv2.VideoCapture(5)

num = 0

cameraMatrix, dist = pickle.load(open( "calibration.pkl", "rb" ))

############## UNDISTORTION #####################################################

# img = cv2.imread('./images/img3.png')
# h,  w = img.shape[:2]
# newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))
h,  w = 480, 640
newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))


while cap.isOpened():
    success, img = cap.read()
    
    

    # img = cv2.undistort(img, cameraMatrix, dist, None, newCameraMatrix)
    # # crop the image
    # x, y, w, h = roi
    # img = img[y:y+h, x:x+w]
    
    cv2.imshow("Img", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()