import face_recognition as fr
import cv2 as cv
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os 

Tk().withdraw()
img = askopenfilename()

target_img = fr.load_image_file(img)
target_encoding = fr.face_encodings(target_img)

print(target_encoding)

def encode_faces(folder):
    list_people_encoding = []

    for filename in os.listdir(folder):
        known_img = fr.load_image_file(f'{folder}{filename}')
        known_encoding = fr.face_encodings(known_img)[0]

        list_people_encoding.append((known_encoding,filename))

    return list_people_encoding


def find_target_face():
    face_location = fr.face_locations(target_img)

    for person in encode_faces('people/'):
        encode_face = person[0]
        filename = person[1]

        is_target_face = fr.compare_faces(encode_face, target_encoding, tolerance=0.6)
        print(f'{is_target_face}{filename}')

        if face_location:
            face_number = 0
            for location in face_location:
                if is_target_face[face_number]:
                    label = filename
                    create_frame(location, label)

                face_number += 1

def create_frame(location, label):
    top, right, bottom, left = location

    cv.rectangle(target_img, (left, top), (right, bottom), (255, 0, 0), 2)
    cv.rectangle(target_img, (left, bottom + 20), (right, bottom), (255, 0, 0), cv.FILLED)
    cv.putText(target_img, label, (left + 3, bottom + 14), cv.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1)

def render_img():
    rgb_img = cv.cvtColor(target_img, cv.COLOR_BGR2RGB)
    cv.imshow('Face Recognition', rgb_img)
    cv.waitKey(0)

find_target_face()
render_img()
