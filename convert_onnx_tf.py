import onnx

# Note that onnx_tf only supports
from onnx_tf.backend import prepare

onnx_model = onnx.load('./u2net.onnx')
tf_rep = prepare(onnx_model)
tf_rep.export_graph('u2net_tf')

print("Model converted to graph successfully")
