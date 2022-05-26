import numpy as np
import cv2
import face_recognition
import os
from datetime import datetime

path= 'imagesattendance'
images=[]
names=[]
myList=os.listdir(path)
print(myList)

for img in myList:
    curImg=cv2.imread(f'{path}/{img}')
    images.append(curImg)
    names.append(os.path.splitext(img)[0])
print(names)

def findEncodings(images):
    encodeList=[]
    for nm in images:
        nm=cv2.cvtColor(nm, cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(nm)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('attendance.csv', 'r+') as f:
        dataList=f.readlines()
        nameList=[]
        for line in dataList:
            entry=line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now=datetime.now()
            dt=now.strftime('%d/%m/%y, %H:%M:%S')
            #dt=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name}, {dt}')
            #f.writelines(f'\n{dt},{name}')


encodeListKnown=findEncodings(images)
print('Encoding Complete')

cap=cv2.VideoCapture(0)

while True:
    success, img= cap.read()
    smallimg=cv2.resize(img,(0,0),None,0.25,0.25)
    smallimg = cv2.cvtColor(smallimg, cv2.COLOR_BGR2RGB)

    facesCurFrame=face_recognition.face_locations(smallimg)
    encodesCurFrame=face_recognition.face_encodings(smallimg,facesCurFrame)

    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches= face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis= face_recognition.face_distance(encodeListKnown,encodeFace)
        matchIndex=np.argmin(faceDis)

        if matches[matchIndex]:
            name=names[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1=faceLoc
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img, (x1,y1), (x2,y2),(255, 0, 255), 3)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (255, 0, 255), cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)

    cv2.imshow('Webcam',img)
    cv2.waitKey(1)
