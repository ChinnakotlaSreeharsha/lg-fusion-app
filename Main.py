import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, Add, Concatenate, LayerNormalization,
    MultiHeadAttention, AveragePooling2D, UpSampling2D
)
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tkinter import Tk, filedialog

# Initialize Tkinter FIRST
root = Tk()
root.withdraw()

print(" Please select the Infrared image file...")
ir_path = filedialog.askopenfilename(title="Select Infrared Image",
                                     filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
print("Infrared image selected:", ir_path)

print("\n Please select the Visible image file...")
vis_path = filedialog.askopenfilename(title="Select Visible Image",
                                      filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
print("Visible image selected:", vis_path)


import matplotlib
matplotlib.use('TkAgg')



# 🔹 Read as COLOR
ir = cv2.imread(ir_path, cv2.IMREAD_COLOR)
vis = cv2.imread(vis_path, cv2.IMREAD_COLOR)

if ir is None or vis is None:
    raise FileNotFoundError(" Image could not be loaded! Please check your selection.")

# 🔹 Resize
ir = cv2.resize(ir, (256, 256))
vis = cv2.resize(vis, (256, 256))

# 🔹 Convert BGR to RGB for correct color display
ir = cv2.cvtColor(ir, cv2.COLOR_BGR2RGB)
vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

# 🔹 Normalize to [0, 1]
ir = ir.astype(np.float32) / 255.0
vis = vis.astype(np.float32) / 255.0

# 🔹 Stack 6 channels (3 from IR + 3 from VIS)
input_data = np.concatenate([ir, vis], axis=-1)
input_data = np.expand_dims(input_data, axis=0)

print("\n Step 1 Done — Color Images Loaded and Normalized Successfully!")



def local_feature_block(x):
    """Extracts local (high-frequency) features."""
    f1 = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    f2 = Conv2D(64, (3,3), activation='relu', padding='same')(f1)
    return f2

def global_feature_block(x):
    """Extracts global contextual features efficiently."""
    norm = LayerNormalization()(x)
    pooled = AveragePooling2D(pool_size=(4, 4))(norm)
    attn = MultiHeadAttention(num_heads=4, key_dim=16)(pooled, pooled)
    upsampled = UpSampling2D(size=(4, 4), interpolation='bilinear')(attn)
    upsampled = Conv2D(x.shape[-1], (1,1), padding='same')(upsampled)
    out = Add()([upsampled, x])
    return out



def fusion_model(input_shape):
    inputs = Input(shape=input_shape)

    local_features = local_feature_block(inputs)
    global_features = global_feature_block(inputs)

    fusion = Concatenate()([local_features, global_features])
    fusion = Conv2D(64, (3,3), activation='relu', padding='same')(fusion)
    output = Conv2D(3, (3,3), activation='sigmoid', padding='same')(fusion)  # RGB output

    return Model(inputs, [local_features, global_features, output])

model = fusion_model((256, 256, 6))
print("Step 2 Done — Color Fusion Model Created Successfully")



local_feat, global_feat, fused = model.predict(input_data)
print(" Step 3 Done — Inference Completed")
vis_ycrcb = cv2.cvtColor(vis, cv2.COLOR_BGR2YCrCb)
Y_vis, Cr, Cb = cv2.split(vis_ycrcb)


alpha = 0.6
ir_gray = cv2.cvtColor((ir * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
ir_gray = ir_gray.astype(np.float32) / 255.0
Y_fused = cv2.addWeighted(Y_vis, 1 - alpha, ir_gray, alpha, 0)


fused_ycrcb = cv2.merge((Y_fused, Cr, Cb))
fused_rgb = cv2.cvtColor(fused_ycrcb, cv2.COLOR_YCrCb2BGR)


os.makedirs("Results_Color", exist_ok=True)

cv2.imwrite("Results_Color/output_infrared.png",
            cv2.cvtColor((ir * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
plt.figure(figsize=(6, 6))
plt.imshow(ir)
plt.title("Infrared Input")
plt.axis("off")
plt.show()


cv2.imwrite("Results_Color/output_visible.png",
            cv2.cvtColor((vis * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
plt.figure(figsize=(6, 6))
plt.imshow(vis)
plt.title("Visible Input")
plt.axis("off")
plt.show()


diff_map = np.mean(ir - vis, axis=-1)
plt.figure(figsize=(6, 6))
plt.imshow(diff_map, cmap='coolwarm')
plt.title("Difference Map")
plt.axis("off")
plt.show()


local_map = np.mean(local_feat[0], axis=-1)
plt.figure(figsize=(6, 6))
plt.imshow(local_map, cmap='gray')
plt.title("Local Feature Map")
plt.axis("off")
plt.show()


global_map = np.mean(global_feat[0], axis=-1)
plt.figure(figsize=(6, 6))
plt.imshow(global_map, cmap='gray')
plt.title("Global Feature Map")
plt.axis("off")
plt.show()

#  Final Fused Image
cv2.imwrite("Results_Color/output_fused_color.png",
            cv2.cvtColor(fused_rgb, cv2.COLOR_RGB2BGR))
plt.figure(figsize=(6, 6))
plt.imshow(fused_rgb)
plt.title("Fused Clear Color Image")
plt.axis("off")
plt.show()


