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

cascade = cv2.CascadeClassifier(cascade_file)

front_face_list = cascade.detectMultiScale(image_gray, minSize = (30, 30))

print(front_face_list)

if len(front_face_list) >= 2:
    (x1, y1, w1, h1) = front_face_list[0]
    (x2, y2, w2, h2) = front_face_list[1]

    img1 = image[y1:y1 +h1, x1:x1 + w1]
    img2 = image[y2:y2 +h2, x2:x2 + w2]
    
    # サイズの調整
    diff_w = int((w1 - w2) / 2)
    diff_h = int((h1 - h2) / 2)

    image_out = image.copy()

    image_out[y1 + diff_h:y1 + diff_h + h2, x1 + diff_w:x1 + diff_w + w2] = img2
    image_out[y2 - diff_h:y2 - diff_h + h1, x2 - diff_w:x2 - diff_w + w1] = img1

    cv2.imshow('image', image_out)
    cv2.waitKey(0)

    cv2.imwrite('./images/out_swap.jpg', image_out)
else:
    print('not detected')
