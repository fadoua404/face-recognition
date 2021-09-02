import numpy as np
import cv2
import pickle

face_cascade=cv2.CascadeClassifier("openCV\src\cascades\data\haarcascade_frontalface_alt2.xml")
#eye_cascade=cv2.CascadeClassifier("openCV\src\cascades\data\haarcascade_eye.xml")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels ={}
with open("labels.pickle", 'rb') as f:
    og_labels=pickle.load(f)
    labels={v:k for k,v in og_labels.items()}

cap=cv2.VideoCapture(0) #This will return video from the first webcam on your computer. (capture)


#This code initiates an infinite loop (to be broken later by a break statement), 
# where we have ret and frame being defined as the cap.read(). 
# Basically, ret is a boolean regarding whether or not there was a return at all, 
# at the frame is each frame that is returned. If there is no frame, you wont get an error, you will get None.
while(True):
    #read frame by frame
    ret , frame = cap.read()
    #the frame, converted to gray 
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    #scaleFactor : Parameter specifying how much the image size is reduced 
    # at each image scale.
    #Parameter specifying how many neighbors each candidate rectangle 
    # should have to retain it. This parameter will affect the quality 
    # of the detected faces: higher value results in less detections 
    # but with higher quality
    for (x,y,w,h) in faces:
        #print(x,y,w,h)
        roi_gray=gray[y:y+h , x:x+w] ##This particular code will return the cropped face from the image
        roi_color=frame[y:y+h , x:x+w]
        #reconize?
        id_ ,conf=recognizer.predict(roi_gray)
        if conf>=30 : # and conf<=100:
            #print(id_)
            #print(labels[id_])
            font=cv2.FONT_HERSHEY_SIMPLEX
            name=labels[id_]
            color=(255,255,255)
            stroke=2
            cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)
        img_item= "my-image.png"
        cv2.imwrite(img_item,roi_gray) #Syntax: cv2.imwrite(filename(must include image format), image)
        
        #draw a rectangle around the face
        color=(255,0,0) #BGR
        stroke=2
        end_cord_x=x+w
        end_cord_y=y+h
        cv2.rectangle(frame , (x,y) , (end_cord_x,end_cord_y),color,stroke)

        #eyes=eye_cascade.detectMultiScale(roi_gray)
        #for (ex,ey,eh,ew) in eyes :
           # cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    #display the resulting frame
    cv2.imshow("frame",frame)
    key=cv2.waitKey(1)  # wait till key press
    if key==ord("q"):  #exit loop on "q" key press # cv2.waitKey(20) & 0xFF ==ord("q"):
        break

#when everything is done ,release the  capture
cap.release()
cv2.destroyAllWindows() #destroy the frame windows

