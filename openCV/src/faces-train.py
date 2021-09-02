import cv2
import os 
from PIL import Image
import numpy as np
import pickle #the process of converting a Python object into a byte stream to store it in a file/database
#The OS module in Python provides functions for interacting with the operating system
#creating and removing a directory (folder), fetching its contents, changing and identifying the current directory, etc

BASE_DIR=os.path.dirname(os.path.abspath(__file__)) #method in Python is used to get the directory name from the specified path.
image_dir=os.path.join(BASE_DIR,"images") #combines one or more path names into a single path

face_cascade=cv2.CascadeClassifier("openCV\src\cascades\data\haarcascade_frontalface_alt2.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id=0 # we want to associate an id to every label
label_ids = {} #dictionnary
y_labels = []
x_train = []

#root : Prints out directories only from what you specified.
#dirs : Prints out sub-directories from root.
#files : Prints out all files from root and directories.

for root , dirs , files , in os.walk(image_dir):
    for file in files :
        if file.endswith("png") or file.endswith("jpg") :
            path = os.path.join(root , file)
            label=os.path.basename(root).replace(" " , "-").lower() 
            print(label,path)

            if not label in label_ids :
                label_ids[label]=current_id
                current_id += 1
            id_=label_ids[label]
            print(label_ids)
            pil_image=Image.open(path).convert("L") # L for luminance : grayscale
            #size=(550,550)
            #final_image=pil_image.resize(size,Image.ANTIALIAS)
            image_array=np.array(pil_image , "uint8")
            #print(image_array)

            faces=face_cascade.detectMultiScale(image_array,scaleFactor=1.5,minNeighbors=5)

            for (x,y,h,w) in faces:
                roi=image_array[y:y+h , x:x+w]
                x_train.append(roi)
                y_labels.append(id_) #a face should have only one id

#print(y_labels)
#print(x_train)

with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids , f)

recognizer.train(x_train,np.array(y_labels))
recognizer.save("trainner.yml")



