import tensorflow as tf

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model('./content/u2net_tf') # path to the SavedModel directory

# IMPORTANT!!!!!!!!!!!!!!!!!
# Dynamic range quantization
# Model from 170MB down to 40MB BUT with a lot of time for inference inside colab and android
# converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# Save the model.
with open('u2net.tflite', 'wb') as f:
  f.write(tflite_model)

