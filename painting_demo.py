import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras


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
                cv2.circle(img,(x,y),10,(255),-1)


    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.circle(img,(x,y),10,(0),-1)
        else:
            cv2.circle(img,(x,y),10,(255),-1)

def prepare_pic(img):
    img = cv2.resize(img,dsize=(50,50))
    img = img/255.0
    img = np.expand_dims(img, axis=[0,3])
    return np.float32(img)

def show_pred(pred):
    path = "letters/{}.jpeg"
    img = cv2.imread(path.format(pred))
    cv2.imshow("classifier", img)

def run_classifier(img):
    x = prepare_pic(img)
    pred = model.predict_classes(x)
    show_pred(pred[0])
    print(pred)

def clean_canvas():
    return np.ones((500,500,1), np.uint8)*255


if __name__ == "__main__":
    img = clean_canvas()
    model = keras.models.load_model("hhd_conv_models/hhd_conv_model_latest")
    cv2.namedWindow('image')
    cv2.namedWindow("classifier")
    cv2.setMouseCallback('image',draw_circle)

    while(1):
        cv2.imshow('image',img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('r'):
            mode = not mode
        elif k == ord('b'):
            img = clean_canvas()
        elif k == ord('d'):
            run_classifier(img)
            img = clean_canvas()
        elif k == 27:
            break

    cv2.destroyAllWindows()