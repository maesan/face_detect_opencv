import cv2
import sys

image_file = None
cascade_file = './haarcascades/haarcascade_frontalface_alt.xml'
mask_file = './images/mask.png'

args = sys.argv

if len(args) > 1:
    image_file = args[1]

image = cv2.imread(image_file)
mask_image = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)

cv2.imshow('image', image)
cv2.waitKey(0)

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cascade = cv2.CascadeClassifier(cascade_file)

front_face_list = cascade.detectMultiScale(image_gray, minSize = (30, 30))

print(front_face_list)

if len(front_face_list):
    for (x,y,w,h) in front_face_list:
        print("[x,y] = %d,%d [w,h] = %d,%d" %(x, y, w, h))

        # 置き換える画像のサイズを決める
        # 長い方の幅に合わせる
        length = w
        if length < h:
            length = h
        
        # ちょっと大きめにして位置調整
        length = int(length * 1.5)
        x = x - int((length - w) / 2)
        y = y - int((length - h) / 2)

        mask_tmp = cv2.resize(mask_image, dsize=(length, length), interpolation=cv2.INTER_LINEAR)

        image[y:length + y, x:length + x] = image[y:length + y, x:length + x] * (1 - mask_tmp[:, :, 3:] / 255) \
                                            + mask_tmp[:, :, :3] * (mask_tmp[:, :, 3:] / 255)
    
    cv2.imwrite('./images/out_replace.jpg', image)
    cv2.imshow('image', image)
    cv2.waitKey(0)
else:
    print('not detected')
