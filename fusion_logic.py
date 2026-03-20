import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, Add, Concatenate, LayerNormalization,
    MultiHeadAttention, AveragePooling2D, UpSampling2D
)

# ---------------- MODEL ---------------- #

def local_feature_block(x):
    f1 = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    f2 = Conv2D(64, (3,3), activation='relu', padding='same')(f1)
    return f2

def global_feature_block(x):
    norm = LayerNormalization()(x)
    pooled = AveragePooling2D(pool_size=(4, 4))(norm)
    attn = MultiHeadAttention(num_heads=4, key_dim=16)(pooled, pooled)
    upsampled = UpSampling2D(size=(4, 4), interpolation='bilinear')(attn)
    upsampled = Conv2D(x.shape[-1], (1,1), padding='same')(upsampled)
    return Add()([upsampled, x])

def fusion_model(input_shape):
    inputs = Input(shape=input_shape)

    local_features = local_feature_block(inputs)
    global_features = global_feature_block(inputs)

    fusion = Concatenate()([local_features, global_features])
    fusion = Conv2D(64, (3,3), activation='relu', padding='same')(fusion)
    output = Conv2D(3, (3,3), activation='sigmoid', padding='same')(fusion)

    return Model(inputs, [local_features, global_features, output])


# ---------------- MAIN FUNCTION ---------------- #

def run_fusion(ir_img, vis_img):

    # Resize
    ir = cv2.resize(ir_img, (256, 256))
    vis = cv2.resize(vis_img, (256, 256))

    # Convert BGR → RGB
    ir = cv2.cvtColor(ir, cv2.COLOR_BGR2RGB)
    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    # Normalize
    ir = ir.astype(np.float32) / 255.0
    vis = vis.astype(np.float32) / 255.0

    # Stack
    input_data = np.concatenate([ir, vis], axis=-1)
    input_data = np.expand_dims(input_data, axis=0)

    # Model
    model = fusion_model((256, 256, 6))

    # Predict
    local_feat, global_feat, fused = model.predict(input_data)

    # Convert back
    vis_ycrcb = cv2.cvtColor(vis, cv2.COLOR_RGB2YCrCb)
    Y_vis, Cr, Cb = cv2.split(vis_ycrcb)

    alpha = 0.6
    ir_gray = cv2.cvtColor((ir * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    ir_gray = ir_gray.astype(np.float32) / 255.0

    Y_fused = cv2.addWeighted(Y_vis, 1 - alpha, ir_gray, alpha, 0)

    fused_ycrcb = cv2.merge((Y_fused, Cr, Cb))
    fused_rgb = cv2.cvtColor(fused_ycrcb, cv2.COLOR_YCrCb2RGB)

    # Convert to uint8 for display
    fused_rgb = (fused_rgb * 255).astype(np.uint8)

    return fused_rgb