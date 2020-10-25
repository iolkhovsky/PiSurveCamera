import cv2
from collections import defaultdict
import glob
import matplotlib.pyplot as plt
from os.path import isdir, join
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm


def load_dataset(root, img_size=(64, 64)):
    assert isdir(root)
    return tf.keras.preprocessing.image_dataset_from_directory(
        root,
        labels="inferred",
        label_mode="int",
        class_names=None,
        color_mode="rgb",
        batch_size=16,
        image_size=img_size,
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation="bilinear",
        follow_links=False,
    )


def save_histogram(histo, hint=None, bins=50):
    assert type(histo) == dict or type(histo) == defaultdict
    if hint is not None:
        assert type(hint) == str
    max_val = max(histo.items(), key=lambda a: a[0])[0]
    min_val = min(histo.items(), key=lambda a: a[0])[0]
    bin_width = int((max_val - min_val) / bins)
    if bin_width == 0:
        bin_width = 1
    plt.bar(histo.keys(), histo.values(), bin_width, color='g')
    filename = "histogram.png" if not hint else "histogram_" + hint + ".png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')


def collect_img_size_statistics(root, save_stat=True):
    imgs = glob.glob(join(root, "**/*.jpg"))
    histo_xsz, histo_ysz = defaultdict(lambda: 0), defaultdict(lambda: 0)
    with tqdm(total=len(imgs), desc=f'Samples ', unit='file') as pbar:
        for idx, img in enumerate(imgs):
            img_ysz, img_xsz = cv2.imread(img, cv2.IMREAD_COLOR).shape[:2]
            histo_xsz[img_xsz] += 1
            histo_ysz[img_ysz] += 1
            pbar.update(1)
    if save_stat:
        save_histogram(histo_xsz, hint="Imgs X size")
        save_histogram(histo_ysz, hint="Imgs Y size")
    return histo_xsz, histo_ysz


if __name__ == "__main__":
    test_root = "/home/igor/datasets/CustomFaces/val"
    collect_img_size_statistics(test_root)
    dataset = load_dataset(root=test_root)
    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(0.1),
        ]
    )
    dataset = dataset.map(
        lambda x, y: (data_augmentation(x, training=True), y))
    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(int(labels[i]))
            plt.axis("off")
    plt.savefig("dataset_samples.png", dpi=300, bbox_inches='tight')
