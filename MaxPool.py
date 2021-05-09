#MaxPool (maxpool_1d_default)
import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare

# Create one Input (ValueInfoProto)
x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 3, 32])


# Create one output (ValueInfoProto)
y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 3, 31])

# Create a node (NodeProto)
node_def = helper.make_node(
    'MaxPool',
    inputs=['x'],
    outputs=['y'],
    kernel_shape=[2],
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [x],
    [y],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-MaxPool')

print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, 'onnx-MaxPool.onnx')
onnx.save(model_def, new_model_path)

print('The model is saved.')



# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, 'onnx-MaxPool.onnx')
onnx_model1 = onnx.load(model_path1)

print('The model is:\n{}'.format(onnx_model1))

import onnxruntime as rt
sess = rt.InferenceSession("onnx-MaxPool.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

x = np.random.randn(1, 3, 32).astype(np.float32)
input_shape = [1, 3, 32]
output_shape = [1, 3, 31]


import skimage.measure
x = np.random.randn(1, 3, 32).astype(np.float32)
y_actual = skimage.measure.block_reduce(x, (1, 1, 1), np.max)

y_pred = sess.run(
        [], {input_name: x})

print("The predicted output for the operation: MaxPool (maxpool_1d_default)", y_pred)

y_pred = np.asarray(y_pred) #converting list into an array
print(y_pred.shape)

y_pred = np.squeeze(y_pred, axis=0)
print(y_pred.shape)
print("y_actual shape: ", y_actual.shape)
y_pred1 = np.round(y_pred, 3)
y_actual1 = np.round(y_actual, 3)
compare(y_pred1, y_actual1)