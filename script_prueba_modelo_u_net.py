

import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from scipy.signal import find_peaks
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import tensorflow as tf


# ---------- Configuración ----------
video_path = r"C:\Users\jandr\Videos\videos prueba\Processed_video\Cold_bogota_down\A7_Cold_Bogota_Elevation_processed.AVI"
frames_dir = r"C:\Users\jandr\Videos\videos prueba\Frames"
masks_dir = r"C:\Users\jandr\Videos\videos prueba\Masks"
masks_output = r"C:\Users\jandr\Videos\videos prueba\masks_output"
frames_output = r"C:\Users\jandr\Videos\videos prueba\Frames_output"

os.makedirs(frames_dir, exist_ok=True)
os.makedirs(masks_dir, exist_ok=True)

img_size = (544, 768)
threshold = 0.5   # umbral para máscara binaria
fps = 25               # frames por segundo de tu video

# Definir la función de pérdida otra vez
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

model_path = r"C:\Users\jandr\OneDrive - Universidad del rosario\Model_segmentation_snail_heart\unet_corazon_caracol.h5"
# ---------- 1️⃣ Extraer frames ----------
cap = cv2.VideoCapture(video_path)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_name = f"frame_{frame_count:04d}.tif"
    cv2.imwrite(os.path.join(frames_output, frame_name), frame)
    frame_count += 1

cap.release()
print(f"{frame_count} frames extraídos")

# ---------- 2️⃣ Cargar modelo ----------
model = load_model(
    model_path,
    custom_objects={
        "dice_loss": dice_loss,
        "dice_bce_loss": dice_bce_loss,
        "dice_coef": dice_coef
    }
)
# ---------- 3️⃣ Predecir máscaras y calcular áreas ----------
areas = []

for filename in sorted(os.listdir(frames_output)):
    if filename.endswith(".tif"):
        # Cargar frame y redimensionar
        frame = load_img(os.path.join(frames_output, filename), target_size=img_size)
        frame_arr = img_to_array(frame) / 255
        frame_arr = np.expand_dims(frame_arr, axis=0)
        
        # Predecir máscara
        pred_mask = model.predict(frame_arr)
        pred_mask_bin = (pred_mask[0,:,:,0] > threshold).astype(np.uint8) * 255
        
        # Guardar máscara
        mask_save_path = os.path.join(masks_output, filename)
        cv2.imwrite(mask_save_path, pred_mask_bin)
        
        # Calcular área en pixeles
        area = np.sum(pred_mask_bin > 0)
        areas.append(area)

# ---------- 4️⃣ Guardar áreas ----------
df = pd.DataFrame({"frame": range(len(areas)), "area_px": areas})
path_csv = r"C:\Users\jandr\OneDrive - Universidad del rosario\Model_segmentation_snail_heart\areas_corazon_A7.csv"
df.to_csv(path_csv, index=False)
print("\Áreas guardadas en areas_corazon.csv")

# ---------- 5️⃣ Estimar frecuencia cardíaca ----------
areas_array = np.array(areas)

# Suavizar señal (media móvil)
window_size = 3
areas_smooth = np.convolve(areas_array, np.ones(window_size)/window_size, mode='same')

# Detectar picos (contracciones)
peaks, _ = find_peaks(areas_smooth, distance=fps*0.25)  # distancia mínima ~0.3s
num_peaks = len(peaks)
duration_sec = len(areas_array) / fps
heart_rate_hz = num_peaks / duration_sec
heart_rate_bpm = heart_rate_hz * 60

print(f"Picos detectados: {num_peaks}")
print(f"Frecuencia cardíaca estimada: {heart_rate_hz:.2f} Hz ({heart_rate_bpm:.1f} bpm)")

