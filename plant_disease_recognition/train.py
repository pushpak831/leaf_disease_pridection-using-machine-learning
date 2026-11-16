"""
Train the plant disease CNN from the command line (same model as Train_plant_disease.ipynb).

    python train.py "C:\\path\\to\\New Plant Diseases Dataset(Augmented)" 10

The first argument is the folder that contains the 'train' and 'valid' folders.
The second (optional) is the number of epochs; the notebook used 10.
Saves 'trained_plant_disease_model.keras' next to this script.

Warning: this model is big. 10 epochs on a normal laptop CPU can take many hours.
Try 1 or 2 epochs first to check that everything works.
"""
import sys
from pathlib import Path

import tensorflow as tf

from plant_labels import CLASS_NAMES, IMG_SIZE, MODEL_FILE_NAMES

HERE = Path(__file__).resolve().parent


def load(folder):
    return tf.keras.utils.image_dataset_from_directory(
        folder, labels="inferred", label_mode="categorical", color_mode="rgb",
        batch_size=32, image_size=(IMG_SIZE, IMG_SIZE), shuffle=True, interpolation="bilinear")


def build_model(n_classes):
    L = tf.keras.layers
    model = tf.keras.models.Sequential()
    model.add(L.Input(shape=(IMG_SIZE, IMG_SIZE, 3)))
    for i, filters in enumerate((32, 64, 128, 256, 512)):
        model.add(L.Conv2D(filters, 3, padding="same", activation="relu"))
        model.add(L.Conv2D(filters, 3, activation="relu"))
        model.add(L.MaxPool2D(pool_size=2, strides=2))
    model.add(L.Dropout(0.25))
    model.add(L.Flatten())
    model.add(L.Dense(1500, activation="relu"))
    model.add(L.Dropout(0.4))
    model.add(L.Dense(n_classes, activation="softmax"))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    if len(sys.argv) < 2:
        sys.exit('Usage: python train.py "<folder containing train and valid>" [epochs]')
    root = Path(sys.argv[1])
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    for sub in ("train", "valid"):
        if not (root / sub).is_dir():
            sys.exit(f"Folder not found: {root / sub}\n"
                     f"The folder you give must contain 'train' and 'valid' inside it.")

    training_set, validation_set = load(root / "train"), load(root / "valid")
    if training_set.class_names != CLASS_NAMES:
        sys.exit("The class folders do not match the 38 expected classes "
                 f"(found {len(training_set.class_names)}). Use the 'New Plant Diseases Dataset'.")

    model = build_model(len(CLASS_NAMES))
    model.summary()
    model.fit(training_set, validation_data=validation_set, epochs=epochs)
    loss, acc = model.evaluate(validation_set)
    print(f"[INFO] Validation accuracy: {acc * 100:.2f}%")
    # Fresh optimizer before saving: the training state is not needed for predictions and
    # leaving it out makes the file about 3x smaller (about 30 MB), easier to share.
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.save(HERE / MODEL_FILE_NAMES[0])
    print(f"[INFO] Saved {MODEL_FILE_NAMES[0]}. Now run: python -m streamlit run main.py")


if __name__ == "__main__":
    main()
