
# import some useful library
import cv2
import numpy as np
import face_recognition as fr
import os
from datetime import datetime

path = 'ImageAttendance'          #  Path for training the model
images = []                       #  These are list for images store
classNames = []
myList = os.listdir(path)         #  it help us to list all the directory from the given path folder
# print(myList)                     #  print the list of the images


## loop for create the class Name list through my list remove the .jpg extension
for cls in myList:
    curImg  = cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])
print(classNames)

## function for find the image encoding convert image colors
def findEncodings(images):
    encodeList = []
    for img in images:
        img= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = fr.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

## function to help for mark the attendance and also
## store the record inside the csv file (comma seperate file)
## i.e Attendance.csv
def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDateList = f.readlines()
        nameList = []
        # print(myDateList)
        for line in myDateList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

#  Calling of the find encoding function
encodeListKnown = findEncodings(images)

# message when encoding completes
print('encoding complete')

## creating a object to access webcam.
## Create an infinite while loop to display each frame of the webcam's video continuously.
cap=cv2.VideoCapture(0)

## loop for predict the attendee to mark the attendance
while True:
    success, img =cap.read()
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    faceCurFrame = fr.face_locations(imgS)
    encodesCurFrame = fr.face_encodings(imgS,faceCurFrame)

    for encodeFace,faceLoc in zip(encodesCurFrame,faceCurFrame):
        matches = fr.compare_faces(encodeListKnown,encodeFace)
        faceDis = fr.face_distance(encodeListKnown,encodeFace)
        # print(faceDis)
        matchIndex=np.argmin(faceDis)

        ##  when it match the attendee with the pretrained record then it will mark it attadance
        if(matches[matchIndex]):
            name=classNames[matchIndex].upper()
            y1,x2,y2,x1=faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2+6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)

    cv2.imshow('webcam',img)


    ## this condition will help us to open the web cam till I pressed the space bar on the keyboard
    ## space bar keyboard key is 32:
    if cv2.waitKey(33) == 32:
        exit(0)  # it will exit the program



                # Thank you















