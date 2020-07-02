import cv2 as cv
import numpy as np
import time
time.sleep(3)

video=cv.VideoCapture(0)


y,background=video.read()
background=np.flip(background,axis=1)

while(1):
    rev,frame=video.read()
    frame=np.flip(frame,axis=1)

    hsv=cv.cvtColor(frame,cv.COLOR_BGR2HSV)

    lower_blue=np.array([94, 80, 2])
    upper_blue=np.array([126, 255, 255])

    mask=cv.inRange(hsv,lower_blue,upper_blue)

    mask=cv.morphologyEx(mask,cv.MORPH_OPEN,np.ones((3,3),np.uint8))
    mask=cv.morphologyEx(mask,cv.MORPH_DILATE,np.ones((3,3),np.uint8))

    mask2=cv.bitwise_not(mask)

    new_image=cv.bitwise_and(frame,frame,mask=mask2)
    new_background=cv.bitwise_and(background,background,mask=mask)

    output=cv.addWeighted(new_image,1,new_background,1,0)



    cv.imshow('frame',output)

    if cv.waitKey(20) & 0xFF==27:
        break

video.release()
cv.destroyAllWindows()