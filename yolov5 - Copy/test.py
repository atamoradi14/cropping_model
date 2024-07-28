import os
import cv2
import numpy as np
import tensorflow as tf

# Load TensorFlow model
model_path = 'crop/retinagpuS/weights/TFModel'  # Replace with actual path
model = tf.saved_model.load(model_path)
print(list(model.signatures.keys()))