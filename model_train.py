import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# --- Config ---
DATA_DIR = "Captcha -20250613T230133Z-1-001\Preprocessed"
IMG_WIDTH, IMG_HEIGHT = 160, 60
CHARS = "0123456789"
MAX_LABEL_LEN = 5
BATCH_SIZE = 32
EPOCHS = 20

# --- Character Mapping ---
char_to_num = keras.layers.StringLookup(vocabulary=list(CHARS), mask_token=None)
num_to_char = keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

# --- Data Loading ---
def load_data():
    images, labels = [], []
    for fname in os.listdir(DATA_DIR):
        if fname.lower().endswith(".png"):
            label = os.path.splitext(fname)[0]
            path = os.path.join(DATA_DIR, fname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            img = img.astype("float32") / 255.0
            img = np.expand_dims(img, axis=-1)
            images.append(img)
            labels.append(label)
    return np.array(images), labels

def encode_labels(labels):
    encoded = []
    for label in labels:
        chars = tf.strings.unicode_split(label, input_encoding="UTF-8")
        encoded.append(char_to_num(chars))
    return keras.preprocessing.sequence.pad_sequences(encoded, maxlen=MAX_LABEL_LEN, padding="post")

def create_dataset(images, labels):
    label_lengths = np.array([len(l) for l in labels])
    inputs = {
        "image": images,
        "label": encode_labels(labels),
        "input_len": np.ones((len(images), 1)) * (IMG_WIDTH // 4),
        "label_len": label_lengths.reshape(-1, 1)
    }
    outputs = np.zeros((len(images), 1))  # dummy output for Keras compatibility
    return tf.data.Dataset.from_tensor_slices((inputs, outputs)).batch(BATCH_SIZE)

# --- Model ---
def build_model():
    input_img = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name="image")
    labels = layers.Input(shape=(MAX_LABEL_LEN,), name="label", dtype="int64")
    input_len = layers.Input(shape=(1,), name="input_len", dtype="int64")
    label_len = layers.Input(shape=(1,), name="label_len", dtype="int64")

    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(input_img)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    new_shape = (IMG_WIDTH // 4, (IMG_HEIGHT // 4) * 64)
    x = layers.Reshape(target_shape=new_shape)(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Dense(len(CHARS) + 1, activation="softmax")(x)

    def ctc_loss_fn(y_true, y_pred):
        return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_len, label_len)

    model = keras.models.Model(inputs=[input_img, labels, input_len, label_len], outputs=x)
    model.add_loss(ctc_loss_fn(labels, x))
    return model

# --- Training ---
if __name__ == "__main__":
    images, labels = load_data()
    x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.1, random_state=42)

    train_ds = create_dataset(x_train, y_train)
    val_ds = create_dataset(x_val, y_val)

    model = build_model()
    model.compile(optimizer="adam")
    model.summary()

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
    model.save("captcha_model_digits.h5")

