import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import itertools

THIN_THRESHOLD = 70  # threshold for the size of the boundinboxes that detected
FRAME_SIZE = 20  # adding white frame with size 20
drawing = False  # true if mouse is pressed
mode = True  # if True, draw rectangle. Press 'm' to toggle to curve
ix, iy = -1, -1
IOU_THRESHOLD = 0.03


# mouse callback function
def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            if mode:
                cv2.circle(img, (x, y), 10, 0, -1)
            else:
                cv2.circle(img, (x, y), 10, 255, -1)


    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode:
            cv2.circle(img, (x, y), 10, 0, -1)
        else:
            cv2.circle(img, (x, y), 10, 255, -1)


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def prepare_pic(img):
    img = cv2.resize(img, dsize=(50, 50))
    img = img / 255.0
    img = np.expand_dims(img, axis=[0, 3])
    return np.float32(img)


def show_pred(pred):
    path = "letters/{}.jpeg"
    img = cv2.imread(path.format(pred))
    cv2.imshow("classifier", img)


def get_letter_img(pred):
    path = "letters/{}.jpeg"
    img = cv2.imread(path.format(pred))
    img = cv2.resize(img, (50, 50))
    return img


def run_classifier(img):
    x = prepare_pic(img)
    pred = model.predict_classes(x)
    show_pred(pred[0])
    print(pred)


def calssify_letter(img):
    x = prepare_pic(img)
    pred = model.predict_classes(x)
    return get_letter_img(pred[0])


def clean_canvas():
    return np.ones((400, 1000, 1), np.uint8) * 255


def relu(y):
    if y - FRAME_SIZE < 0:
        return 0
    else:
        return y - FRAME_SIZE


def get_overlap_bb(bb1, bb2):
    x1 = min(bb1[0], bb2[0])
    y1 = min(bb1[1], bb2[1])
    x2 = max(bb1[2], bb2[2])
    y2 = max(bb1[3], bb2[3])
    return [x1, y1, x2, y2]


def get_letters_from_words(img):
    contours, hierarchy = cv2.findContours(cv2.bitwise_not(img), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    word_img = []
    roi_list = []
    bb_list = []
    bad_idx = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        bb = [x, y, x + w, y + h]
        if h > THIN_THRESHOLD or w > THIN_THRESHOLD:
            bb_list.append(bb)
            # roi = img[relu(y):y + h + FRAME_SIZE, relu(x):x + w + FRAME_SIZE]
            # # word_img.append(calssify_letter(roi))
            # roi_list.append(cv2.resize(roi, (50, 50)))
    combinations = list(itertools.combinations(range(len(bb_list)), 2))
    for idx1, idx2 in combinations:
        iou = bb_intersection_over_union(bb_list[idx1], bb_list[idx2])
        print("the iou is {} and the index are ({},{})".format(iou, idx1, idx2))
        if iou > IOU_THRESHOLD:
            new_bb = get_overlap_bb(bb_list[idx1], bb_list[idx2])
            bad_idx.append(idx1)
            bad_idx.append(idx2)
            bb_list.append(new_bb)
    bb_list = np.array(bb_list)
    good_idx = list(set(range(len(bb_list))) - set(bad_idx))
    bb_list = list(bb_list[good_idx])
    bb_list.sort(key=lambda box: box[0])
    for bb in bb_list:
        x, y, x2, y2 = bb
        roi = img[relu(y):y2 + FRAME_SIZE, relu(x):x2 + FRAME_SIZE]
        # word_img.append(calssify_letter(roi))
        roi_list.append(cv2.resize(roi, (50, 50)))

    if word_img:
        print("pass detection for now")
        # word = np.concatenate(word_img, axis=1)
        # cv2.imshow("classifier", word)
    else:
        print("failed")
    if roi_list:
        roi_word = np.concatenate(roi_list, axis=1)
        cv2.imshow("debug", roi_word)


if __name__ == "__main__":
    img = clean_canvas()
    model = keras.models.load_model("hhd_conv_models/hhd_conv_model_latest")
    cv2.namedWindow('image')
    cv2.namedWindow("classifier")
    cv2.namedWindow('debug')
    cv2.setMouseCallback('image', draw_circle)

    while (1):
        cv2.imshow('image', img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('r'):
            mode = not mode
        elif k == ord('b'):
            img = clean_canvas()
        elif k == ord('d'):
            run_classifier(img)
        elif k == ord('w'):
            get_letters_from_words(img)
            img = clean_canvas()
        elif k == 27:
            break

    cv2.destroyAllWindows()
