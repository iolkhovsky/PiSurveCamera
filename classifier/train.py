import cv2
from glob import glob
from os.path import join
from tensorflow import keras
from tensorflow.keras import layers

from classifier.dataset import load_dataset
from classifier.model import make_classifier
from classifier.inference import predict_signle_image


def train(dataset_root, image_size=(64, 64), epochs=5, classes_cnt=2):
    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(0.1),
        ]
    )
    train_dset_root = join(dataset_root, "train")
    val_dset_root = join(dataset_root, "val")

    train_ds = load_dataset(train_dset_root)
    val_ds = load_dataset(val_dset_root)

    train_ds = train_ds.map(
      lambda x, y: (data_augmentation(x, training=True), y))
    val_ds = val_ds.map(
      lambda x, y: (data_augmentation(x, training=False), y))

    train_ds = train_ds.prefetch(buffer_size=32)
    val_ds = val_ds.prefetch(buffer_size=32)

    model = make_classifier(input_shape=image_size + (3,), num_classes=classes_cnt)
    #keras.utils.plot_model(model, show_shapes=True)

    callbacks = [
        keras.callbacks.ModelCheckpoint("checkpoints/model_ep{epoch}.h5"),
    ]

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy" if classes_cnt == 2 else "categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
    )
    return model


if __name__ == "__main__":
    dataset_path = "/home/igor/datasets/CustomFaces/"
    clf = train(dataset_root=dataset_path, epochs=10)
    val_path = join(dataset_path, "val")
    img_paths = glob(join(val_path, "**/*.jpg"))
    img_size = (64, 64)
    for idx, impath in enumerate(img_paths):
        img = cv2.imread(impath, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        prediction = predict_signle_image(clf, img, {0: "Female", 1: "Male"})
        print(idx, impath, prediction)
