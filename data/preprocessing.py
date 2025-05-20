import cv2
import numpy as np
import glob as gb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight


def dull_razor(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)
    result = cv2.inpaint(image, mask, inpaintRadius=1, flags=cv2.INPAINT_TELEA)
    return result

def load_and_prepare_data(img_dir_yes, img_dir_no, img_size=(256, 256)):
    """load_and_prepare_data(img_dir_yes,img_dir_no, img_size=(default 256,256))"""
    paths_yes = gb.glob(img_dir_yes)
    paths_no = gb.glob(img_dir_no)
    data = []
    number = 0

    for label, paths in zip(["yes", "no"], [paths_yes, paths_no]): 
        for path in paths:
            image = cv2.imread(path) 
            if image is None: 
                continue
            image = dull_razor(image)
            image = cv2.resize(image, img_size)
            pixels = image.flatten().tolist()
            number += 1
            print(number)


            data.append([label, pixels])

    df = pd.DataFrame(data, columns=["label", "pixels"])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True) 


    encoder = LabelEncoder()
    df["label"] = encoder.fit_transform(df["label"])

    X = np.stack(df["pixels"].values).reshape(-1, *img_size, 3) 
    y = df["label"].values 

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, stratify=y_train_val, random_state=42)

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y), 
        y=y 
    ) 
    class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

    X_train = X_train / 255.0
    X_val = X_val / 255.0
    X_test = X_test / 255.0

    return X_train, X_val, X_test, y_train, y_val, y_test, class_weights_dict


def Website_foto(image_path, img_size=(256, 256)):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Negalima nuskaityti paveikslelio: {image_path}")

    image = dull_razor(image)
    image = cv2.resize(image, img_size)
    image = image.astype('float32') / 255.0 

    image = np.expand_dims(image, axis=0) 
    return image


