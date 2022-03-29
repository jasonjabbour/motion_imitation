import tensorflow as tf

path = "bittle_frozen_axis1.pb"

converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(path, input_arrays=['input/Ob'], output_arrays=['chicken'])
tflite_model = converter.convert()

tflite_save_file = "bittle_frozen_axis1.tflite"
with open(tflite_save_file, 'wb') as f:
        f.write(tflite_model)

print("Model Converted to TFLite")