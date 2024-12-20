import tensorflow as tf

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="./content/model_float32.tflite")
interpreter.allocate_tensors()

# Get all tensor details
tensor_details = interpreter.get_output_details()
total_tensors = len(tensor_details)

# Print the total number of tensors
print("Total number of tensors:", total_tensors)

# Print input details
input_details = interpreter.get_input_details()
print("Input Details:", input_details)

