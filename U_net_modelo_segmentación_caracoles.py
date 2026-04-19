import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

# ---------- Configuración ----------
frames_dir = r"C:\Users\jandr\OneDrive - Universidad del rosario\Gui_xylem\ROIs"
masks_dir = r"C:\Users\jandr\OneDrive - Universidad del rosario\Gui_xylem\Masks"

img_size = (544, 768)
batch_size = 8
epochs = 50

def load_images(img_dir, mask=False):
    imgs = []
    for filename in sorted(os.listdir(img_dir)):
        if filename.endswith(".tif") or filename.endswith(".png"):
            img = load_img(
                os.path.join(img_dir, filename),
                target_size=img_size,
                color_mode="rgb" if not mask else "grayscale"
            )
            img = img_to_array(img)

            if mask:
                img = (img > 127).astype(np.float32)
            else:
                img = img / 255.0

            imgs.append(img)
    return np.array(imgs)

# ---------- Función para cargar imágenes ----------
def load_images(img_dir, mask=False):
    imgs = []
    for filename in sorted(os.listdir(img_dir)):
        if filename.endswith(".tif") or filename.endswith(".png"):
            img = load_img(os.path.join(img_dir, filename),
                           target_size=img_size,
                           color_mode="rgb" if not mask else "grayscale")
            img = img_to_array(img)

            if mask:
                # ⚠️ binarizar estrictamente
                img = (img > 0).astype(np.float32)
            else:
                img = img / 255.0

            imgs.append(img)
    return np.array(imgs)

# ---------- Cargar datos ----------
X = load_images(frames_dir)
Y = load_images(masks_dir, mask=True)

# Asegurar forma (h,w,1)
if len(Y.shape) == 3:
    Y = np.expand_dims(Y, axis=-1)

# Split train/val
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# ---------- U-Net ----------
def unet(input_size=(256,256,3)):
    inputs = layers.Input(input_size)

    # Encoder
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2,2))(c1)

    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2,2))(c2)

    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2,2))(c3)

    # Bottleneck
    c4 = layers.Conv2D(128, 3, activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(128, 3, activation='relu', padding='same')(c4)

    # Decoder
    u5 = layers.UpSampling2D((2,2))(c4)
    u5 = layers.concatenate([u5, c3])
    c5 = layers.Conv2D(64, 3, activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(64, 3, activation='relu', padding='same')(c5)

    u6 = layers.UpSampling2D((2,2))(c5)
    u6 = layers.concatenate([u6, c2])
    c6 = layers.Conv2D(32, 3, activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(32, 3, activation='relu', padding='same')(c6)

    u7 = layers.UpSampling2D((2,2))(c6)
    u7 = layers.concatenate([u7, c1])
    c7 = layers.Conv2D(16, 3, activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(16, 3, activation='relu', padding='same')(c7)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c7)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

# ---------- Métrica / Loss ----------
def dice_loss(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2.*intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_bce_loss(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    dice = dice_loss(y_true, y_pred)
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return dice + bce

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.*intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


# ---------- Compilar ----------

model = unet(input_size=(img_size[0], img_size[1], 3))
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=dice_bce_loss,
    metrics=[dice_coef]
)
model.summary()

# ---------- Entrenar ----------
model.fit(X_train, Y_train,
          validation_data=(X_val, Y_val),
          batch_size=batch_size,
          epochs=epochs)

# ---------- Guardar ----------
model.save("xylem_segmentation.h5")
print(" Modelo guardado: xylem_segmentation.h5")
