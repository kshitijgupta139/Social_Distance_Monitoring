import cv2
import imutils

#image=cv2.imread('kshitij.jpg')    #reading frames from image file
cap=cv2.VideoCapture('test_video.mp4')      #VideoCapture is used to read frame from video
while True:
    ret,frame=cap.read()   #ret=true if we are able to read frames from video file else false and all the frames are stores in frame variable
    frame=imutils.resize(frame,width=800)
    text="This is my custom text"
    cv2.putText(frame,text,(5,30),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,1,(0,0,255),1)
    cv2.rectangle(frame,(50,50),(500,500),(0,0,255),2)
    cv2.imshow('Application', frame)

    key=cv2.waitKey(1)
    if key==ord('q'):
        break
cv2.destroyAllWindows()