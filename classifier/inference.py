import cv2
from glob import glob
from tensorflow import keras
import tensorflow as tf
import numpy as np
from os.path import join


def predict_signle_image(model, img, labels_map=None):
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)[0]
    if labels_map is None:
        return predictions
    else:
        out = {}
        for idx, p in enumerate(predictions):
            label = str(idx)
            if idx in labels_map.keys():
                label = labels_map[idx]
            out[label] = predictions[idx]
        return out


if __name__ == "__main__":
    loaded_model = keras.models.load_model("save_at_5.h5")
    dataset_path = "/home/igor/datasets/CustomFaces/"
    val_path = join(dataset_path, "train")
    img_paths = glob(join(val_path, "**/*.jpg"))
    img_size = (64, 64)
    for idx, impath in enumerate(img_paths):
        img = cv2.cvtColor(cv2.imread(impath, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        prediction = predict_signle_image(loaded_model, img, {0: "Female", 1: "Male"})
        print(idx, impath, prediction)
