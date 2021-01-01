import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random


drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1

# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv2.circle(img,(x,y),10,(0),-1)
            else:
                cv2.rectangle(img,(ix,iy),(x,y),(0),-1)


    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.circle(img,(x,y),10,(0),-1)
        else:
            cv2.rectangle(img,(ix,iy),(x,y),(0),-1)

def prepare_pic(img):
    img = cv2.resize(img,dsize=(50,50))
    img = img/255.0
    img = np.expand_dims(img, axis=[0,3])
    return np.float32(img)

def show_letter(pred):
    path = "letters/{}.jpeg"
    img = cv2.imread(path.format(pred))
    cv2.imshow("classifier", img)

def run_classifier(img):
    x = prepare_pic(img)
    pred = model.predict_classes(x)
    show_letter(pred[0])
    print(pred)

def check_drawing(img, gt):
    x = prepare_pic(img)
    pred = model.predict_classes(x)
    print(gt,pred[0])
    if gt == pred[0]:
        print("great_job")
        letter = generate_letter()
        show_letter(letter)
        return letter
    else:
        print("keep_trying")
        return gt

def clean_canvas():
    return np.ones((500,500,1), np.uint8)*255

def generate_letter():
    return random.randint(0,26)




if __name__ == "__main__":
    img = clean_canvas()
    model = keras.models.load_model("hdd_conv/hhd_conv_model_latest")
    cv2.namedWindow('image')
    cv2.namedWindow("classifier")
    cv2.setMouseCallback('image',draw_circle)
    letter = generate_letter()

    while(1):
        show_letter(letter)
        cv2.imshow('image',img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('m'):
            mode = not mode
        elif k == ord('b'):
            img = clean_canvas()
        elif k == ord('d'):
            letter = check_drawing(img,letter)
            img = clean_canvas()
        elif k == ord('n'):
            letter = generate_letter()
            show_letter(letter)
            img = clean_canvas() 
        elif k == 27:
            break

    cv2.destroyAllWindows()