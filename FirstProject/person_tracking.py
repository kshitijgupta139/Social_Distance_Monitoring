import cv2
import datetime
import imutils
import numpy as np
from centroidtracker import CentroidTracker

protopath = "MobileNetSSD_deploy.prototxt"
modelpath = "MobileNetSSD_deploy.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

tracker=CentroidTracker(maxDisappeared=80,maxDistance=90) #maxDisappeared=

def non_max_suppression_fast(boxes, overlapThresh):
    try:
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")
    except Exception as e:
        print("Exception occurred in non_max_suppression : {}".format(e))

def main():
    cap=cv2.VideoCapture('test_video.mp4')

    fps_start_time=datetime.datetime.now()
    fps=0
    total_frames=0

    while True:
         ret,frame=cap.read()  #reading the frames
         frame=imutils.resize(frame,width=300)      #resizing the frame
         total_frames=total_frames+1

         (H,W)=frame.shape[:2]
         blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)  #converting frame to blob

         detector.setInput(blob)  #passing blob to detector
         person_detections = detector.forward()  # contains all the detections from our model file
         rects=[]

         for i in np.arange(0, person_detections.shape[2]):   #in this for loop we are getting all the detections i.e coordinates
             confidence = person_detections[0, 0, i, 2]
             if confidence > 0.6:
                 idx = int(person_detections[0, 0, i, 1])  # use to identify belongs to which class

                 if CLASSES[idx] != "person":
                     continue

                 person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                 (startX, startY, endX, endY) = person_box.astype("int")
                 rects.append(person_box)    #will append all the coordinates of the person which our code is detecting
         boundingboxes=np.array(rects)
         boundingboxes=boundingboxes.astype(int)
         rects=non_max_suppression_fast(boundingboxes,0.3)  #used to remove noise from frame

         objects=tracker.update(rects)  #once it accepts all the coordinates it gives us the object id
         for (objectId,bbox) in objects.items():
             x1,y1,x2,y2=bbox
             x1=int(x1)
             y1=int(y1)
             x2=int(x2)
             y2=int(y2)

             cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
             text="ID:{}".format(objectId)  #used to display object id
             cv2.putText(frame,text,(x1,y1-5),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1)

         fps_end_time=datetime.datetime.now()
         time_diff=fps_end_time-fps_start_time
         if time_diff.seconds==0:
            fps=0.0
         else:
             fps=(total_frames/time_diff.seconds)
         fps_text= "FPS: {:.2f}".format(fps)
         cv2.putText(frame,fps_text,(5,30),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1)
         cv2.imshow('Application',frame)
         key=cv2.waitKey(1)
         if key==ord('q'):
             break
    cv2.destroyAllWindows()

main()