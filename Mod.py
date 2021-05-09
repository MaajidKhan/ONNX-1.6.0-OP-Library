#Mod (mod_broadcast)
import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare

# Create one Input (ValueInfoProto)
x = helper.make_tensor_value_info('x', TensorProto.INT32, [3, 2, 5])
y = helper.make_tensor_value_info('y', TensorProto.INT32, [1])

# Create one output (ValueInfoProto)
z = helper.make_tensor_value_info('z', TensorProto.INT32, [3, 2, 5])

# Create a node (NodeProto)
node_def = helper.make_node(
    'Mod',
    inputs=['x', 'y'],
    outputs=['z'],
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [x, y],
    [z],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-Mod')

print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, 'onnx-Mod.onnx')
onnx.save(model_def, new_model_path)

print('The model is saved.')



# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, 'onnx-Mod.onnx')
onnx_model1 = onnx.load(model_path1)

print('The model is:\n{}'.format(onnx_model1))

import onnxruntime as rt
sess = rt.InferenceSession("onnx-Mod.onnx")
input_name = sess.get_inputs()[0].name
input_name1 = sess.get_inputs()[1].name
label_name = sess.get_outputs()[0].name

x = np.arange(0, 30).reshape([3, 2, 5])
y = np.array([7])
y_actual = np.mod(x, y)
#   array([[[0, 1, 2, 3, 4],
#     [5, 6, 0, 1, 2]],

#    [[3, 4, 5, 6, 0],
#     [1, 2, 3, 4, 5]],

#    [[6, 0, 1, 2, 3],
#     [4, 5, 6, 0, 1]]], dtype=int32)

y_pred = sess.run(
        [], {input_name: x,
        input_name1: y})

print("The predicted output for the operation: Mod (mod_broadcast)", y_pred)

y_pred = np.asarray(y_pred) #converting list into an array
print(y_pred.shape)

y_pred = np.squeeze(y_pred, axis=0)
print(y_pred.shape)

compare(y_pred, y_actual)

