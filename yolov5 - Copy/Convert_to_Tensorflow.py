import onnx
from onnx_tf.backend import prepare

# Load the ONNX model
onnx_model = onnx.load('crop/retinagpuS/weights/best.onnx')  # Adjust the path as needed

# Convert the ONNX model to TensorFlow
tf_rep = prepare(onnx_model)

# Export the TensorFlow model
tf_rep.export_graph('crop/retinagpuS/weights/TFModel')  # Adjust the path as needed
