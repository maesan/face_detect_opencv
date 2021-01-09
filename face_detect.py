import cv2
import sys

image_file = None
cascade_file = './haarcascades/haarcascade_frontalface_alt.xml'

args = sys.argv

if len(args) > 1:
    image_file = args[1]

image = cv2.imread(image_file)

cv2.imshow('image', image)
cv2.waitKey(0)

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow('image', image_gray)
cv2.waitKey(0)

cascade = cv2.CascadeClassifier(cascade_file)

front_face_list = cascade.detectMultiScale(image_gray, minSize = (30, 30))

print(front_face_list)

if len(front_face_list):
    for (x,y,w,h) in front_face_list:
        print("[x,y] = %d,%d [w,h] = %d,%d" %(x, y, w, h))
        cv2.rectangle(image, (x,y), (x+w, y+h), (0, 0, 255), thickness=10)
    
    cv2.imshow('image', image)
    cv2.waitKey(0)
else:
    print('not detected')
