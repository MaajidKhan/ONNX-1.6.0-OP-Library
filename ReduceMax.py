#ReduceMax(default_axes_keepdims)

import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare

# The protobuf definition can be found here:
# https://github.com/onnx/onnx/blob/master/onnx/onnx.proto

data = helper.make_tensor_value_info('data', TensorProto.FLOAT, [3, 2, 2])

# Create one output (ValueInfoProto)
reduced = helper.make_tensor_value_info('reduced', TensorProto.FLOAT, [1, 1, 1])

shape = [3, 2, 2]
axes = None
keepdims = 1

# Create a node (NodeProto)
node_def = helper.make_node(
    'ReduceMax',
    inputs=['data'],
    outputs=['reduced'],
    keepdims=keepdims
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [data],
    [reduced],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-ReduceMax')

print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, 'onnx-ReduceMax.onnx')
onnx.save(model_def, new_model_path)

print('The model is saved.')



# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, 'onnx-ReduceMax.onnx')
onnx_model1 = onnx.load(model_path1)

print('The model is:\n{}'.format(onnx_model1))



import onnxruntime as rt
import numpy as np
sess = rt.InferenceSession("onnx-ReduceMax.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)
y_actual = np.maximum.reduce(data, axis=axes, keepdims=keepdims == 1)
#print(reduced)
[[[60.]]]
# print(reduced)
# [[[60.00671387]]]

y_pred = sess.run(
        [label_name], {input_name: data
        })

print("The predicted output for the operation: ReduceMax (default_axes_keepdims)")
print(y_pred)

y_pred = np.asarray(y_pred) #converting list into an array
print(y_pred.shape)

y_pred = np.squeeze(y_pred, axis=0)
print(y_pred.shape)

compare(y_actual, y_pred)