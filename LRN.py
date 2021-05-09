#LRN (default)
import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare
import math

# Create one Input (ValueInfoProto)
x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [5, 5, 5, 5])

# Create one output (ValueInfoProto)
y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [5, 5, 5, 5])

alpha = 0.0001
beta = 0.75
bias = 1.0
nsize = 3

# Create a node (NodeProto)
node_def = helper.make_node(
    'LRN',
    inputs=['x'],
    outputs=['y'],
    size=3
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [x],
    [y],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-LRN')

print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, 'onnx-LRN.onnx')
onnx.save(model_def, new_model_path)

print('The model is saved.')



# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, 'onnx-LRN.onnx')
onnx_model1 = onnx.load(model_path1)

print('The model is:\n{}'.format(onnx_model1))

import onnxruntime as rt
sess = rt.InferenceSession("onnx-LRN.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

x = np.random.randn(5, 5, 5, 5).astype(np.float32)
square_sum = np.zeros((5, 5, 5, 5)).astype(np.float32)
for n, c, h, w in np.ndindex(x.shape):
    square_sum[n, c, h, w] = sum(x[n,
                                   max(0, c - int(math.floor((nsize - 1) / 2))):min(5, c + int(math.ceil((nsize - 1) / 2)) + 1),
                                   h,
                                   w] ** 2)
y_actual = x / ((bias + (alpha / nsize) * square_sum) ** beta)

y_pred = sess.run(
        [], {input_name: x
       })

print("The predicted output for the operation: LRN", y_pred)

y_pred = np.asarray(y_pred) #converting list into an array
print(y_pred.shape)

y_pred = np.squeeze(y_pred, axis=0)
print(y_pred.shape)

y_pred1 = np.round(y_pred, 3)
y_actual1 = np.round(y_actual, 3)
compare(y_pred1, y_actual1)
