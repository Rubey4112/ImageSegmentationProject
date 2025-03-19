import cv2

cap = cv2.VideoCapture(5)

num = 0

while cap.isOpened():
    success, img = cap.read()

    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord("s"):
        cv2.imwrite("images_charuco/img" + str(num) + ".png", img)
        print("image saved")
        num += 1

    cv2.imshow("Img", img)

cap.release()