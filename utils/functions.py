## import libraries

import numpy as np
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

## 

# create function to remove layers, remove whitespace, and resize images before adding them to the dataframe
def prepare_image(path):
    layer = cv2.split(cv2.imread(path))[0]
    new_rows = []
    for row in layer:
        if row.mean() != 255:
            new_rows.append(row)
    new_rows = np.array(new_rows)
    new_columns = []
    for column in new_rows.T:
        if column.mean() != 255:
            new_columns.append(column)
    new_array = np.array(new_columns).T
    # the image should be a square - add whitespace to make it into a square
    width = new_array.shape[1]
    height = new_array.shape[0]
    if width > height+3:
        # add 3 in order to account for images which are already close enough to being squares
        amount_rows_to_add = (width-height)//2
        row_to_add = [255.0 for n in range(width)]
        rows_to_add = np.array([row_to_add for n in range(amount_rows_to_add)])
        new_array = np.vstack([rows_to_add, new_array, rows_to_add])
    if height > width+3:
        transpose = new_array.T
        amount_rows_to_add = (height-width)//2
        row_to_add = [255.0 for n in range(height)]
        rows_to_add = np.array([row_to_add for n in range(amount_rows_to_add)])
        new_array = np.vstack([rows_to_add, transpose, rows_to_add]).T
    return cv2.resize(new_array, (40, 40))

def prepare_labels_for_model(original_labels):
    y, label_index = pd.factorize(original_labels)
    return y, label_index

def prepare_data_for_model(X, labels):
    y, label_index = pd.factorize(labels)
    x_train, x_test, y_train, y_test = train_test_split([val[0] for val in X.values], y, test_size = 0.25, stratify=y, random_state=22)
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2],1)
    num_labels = len(set(label_index))
    y_training = to_categorical(y_train, num_labels = num_labels, dtype='int')
    y_testing = to_categorical(y_test, num_labels = num_labels, dtype='int')
    x_train_norm = x_train/255
    x_test_norm = x_test/255
    x_train_norm_train, x_validate, y_training_train, y_validate = train_test_split(x_train_norm, 
                    y_training, test_size = 0.20, stratify=y_training, random_state=22)
    return x_train_norm_train, x_validate, x_test_norm, y_training_train, y_validate, y_testing, label_index

def convert_to_BW(path):
    layer = cv2.split(cv2.imread(path))[0]
    BW_image = []
    for row in layer:
        new_row = [0 if val < 140 else 255 for val in row]
        BW_image.append(new_row)
    BW_image = np.array(BW_image)
    return BW_image

def split_letters_page(path):
    img = cv2.imread(path)

    lower = (0, 80, 110)
    upper = (0, 120, 150)

    mask = cv2.inRange(img, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    img[mask==255] = (255,255,255)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU )[1] 

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17,17))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    morph = 255 - morph

    bboxes = []
    bboxes_img = img.copy()
    contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    for cntr in contours:
        x,y,w,h = cv2.boundingRect(cntr)
        cv2.rectangle(bboxes_img, (x, y), (x+w, y+h), (0, 0, 255), 1)
        bboxes.append((x,y,w,h))

    def takeFirst(elem):
        return elem[0]

    bboxes.sort(key=takeFirst)

    crops = []

    for i in range(len(bboxes)):
        (x,y,w,h) = bboxes[i]
        crop = img[y-10:y+h+10, x-10:x+w+10]
        crops.append(crop)
    return bboxes_img, crops
