import cv2
import numpy as np
import glob as gb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight




def load_and_prepare_data(img_dir_yes, img_dir_no, img_size=(256, 256)):
    """load_and_prepare_data(img_dir_yes,img_dir_no, img_size=(default 256,256))"""
    paths_yes = gb.glob(img_dir_yes) #Naudojamas glob (failų paieškos funkcija) – suranda visus atitinkančius failus iš img_dir_yes ir img_dir_no katalogų.
    paths_no = gb.glob(img_dir_no)
    data = []
    number = 0

    for label, paths in zip(["yes", "no"], [paths_yes, paths_no]): #kuris eina per dvi sąrašų poras vienu metu
        for path in paths: #kiekvienas failo kelias apdorojamas atskirai.
            image = cv2.imread(path) #Įkeliamas paveikslėlis iš disko.
            if image is None: #Jei paveikslėlio nepavyko įkelti (gal failas sugadintas), pereina prie kito.
                continue
            image = cv2.resize(image, img_size)
            pixels = image.flatten().tolist()#Paveikslėlio trimačiai pikselių duomenys (aukštis x plotis x spalvų kanalai) paverčiami į vienmatį vektorių.
            number += 1
            print(number)


            data.append([label, pixels])

    df = pd.DataFrame(data, columns=["label", "pixels"])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True) #– Atsitiktinai sumaišomi duomenys (shuffle), kad modelis nemokytųsi šališkai. random_state=42 garantuoja pakartojamumą.


    encoder = LabelEncoder()
    df["label"] = encoder.fit_transform(df["label"])

    X = np.stack(df["pixels"].values).reshape(-1, *img_size, 3) #Visi paveikslėlių vektoriai sujungiami į vieną numpy masyvą
    y = df["label"].values #Ištraukiame etikečių reikšmes į y masyvą

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, stratify=y_train_val, random_state=42)

    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y) #automatiškai apskaičiuoja klasių svorius, kurie kompensuoja disbalansą tarp klasių.
    class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

    return X_train / 255.0, X_val / 255.0, X_test / 255.0, y_train, y_val, y_test, class_weights_dict

