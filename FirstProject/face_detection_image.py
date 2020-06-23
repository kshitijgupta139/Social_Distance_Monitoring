import cv2
import numpy as np
import imutils

protopath = "deploy.prototxt"
modelpath = "res10_300x300_ssd_iter_140000.caffemodel"   #model trained to detect only faces
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)




def main():
    image = cv2.imread('people.jpg')
    image = imutils.resize(image, width=500)

    (H, W) = image.shape[:2]

    face_blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), False,False)

    detector.setInput(face_blob)
    face_detections = detector.forward()   #contains all the detections from our model file

    for i in np.arange(0, face_detections.shape[2]):
        confidence = face_detections[0, 0, i, 2]
        if confidence > 0.15:

            face_box = face_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = face_box.astype("int")

            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)

    cv2.imshow("Results", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()