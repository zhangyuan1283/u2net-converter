import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="./content/u2net.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
print("Input Details:", input_details)
