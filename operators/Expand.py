#Expand (dim_changed)
import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare

# Create one Input (ValueInfoProto)
data = helper.make_tensor_value_info('data', TensorProto.FLOAT, [3, 1])
new_shape = helper.make_tensor_value_info('new_shape', TensorProto.INT64, [3])

# Create one output (ValueInfoProto)
expanded = helper.make_tensor_value_info('expanded', TensorProto.FLOAT, [2, 3, 6])


# Create a node (NodeProto)
node_def = helper.make_node(
    'Expand',
    inputs=['data', 'new_shape'],
    outputs=['expanded'],
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [data, new_shape],
    [expanded],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-Expand')

print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, '../onnx_generated_models/onnx-Expand.onnx')
onnx.save(model_def, new_model_path)

print('The model is saved.')



# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, '../onnx_generated_models/onnx-Expand.onnx')
onnx_model1 = onnx.load(model_path1)

print('The model is:\n{}'.format(onnx_model1))

import onnxruntime as rt
sess = rt.InferenceSession("../onnx_generated_models/onnx-Expand.onnx")
input_name = sess.get_inputs()[0].name
input_name1 = sess.get_inputs()[1].name
label_name = sess.get_outputs()[0].name

shape = [3, 1]
data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
#print(data)
#[[1.], [2.], [3.]]
new_shape = [2, 1, 6]
expanded = data * np.ones(new_shape, dtype=np.float32)
#print(expanded)
#[[[1., 1., 1., 1., 1., 1.],
#  [2., 2., 2., 2., 2., 2.],
#  [3., 3., 3., 3., 3., 3.]],
#
# [[1., 1., 1., 1., 1., 1.],
#  [2., 2., 2., 2., 2., 2.],
#  [3., 3., 3., 3., 3., 3.]]]
new_shape = np.array(new_shape, dtype=np.int64)

y_pred = sess.run(
        [], {input_name: data,
        input_name1: new_shape})

print("The predicted output for the operation: Expand", y_pred)

y_pred = np.asarray(y_pred) #converting list into an array
print(y_pred.shape)

y_pred = np.squeeze(y_pred, axis=0)
print(y_pred.shape)

compare(y_pred, expanded)

