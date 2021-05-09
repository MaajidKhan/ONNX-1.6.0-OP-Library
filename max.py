#max
import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare

# Create one Input (ValueInfoProto)
data_0 = helper.make_tensor_value_info('data_0', TensorProto.FLOAT, [3])
data_1 = helper.make_tensor_value_info('data_1', TensorProto.FLOAT, [3])
data_2 = helper.make_tensor_value_info('data_2', TensorProto.FLOAT, [3])

# Create one output (ValueInfoProto)
result = helper.make_tensor_value_info('result', TensorProto.FLOAT, [3])

# Create a node (NodeProto)
node_def = helper.make_node(
    'Max',
    inputs=['data_0', 'data_1', 'data_2'],
    outputs=['result'],
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [data_0, data_1, data_2],
    [result],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-max')

print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, 'onnx-max.onnx')
onnx.save(model_def, new_model_path)

print('The model is saved.')



# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, 'onnx-max.onnx')
onnx_model1 = onnx.load(model_path1)

print('The model is:\n{}'.format(onnx_model1))

import onnxruntime as rt
sess = rt.InferenceSession("onnx-max.onnx")
input_name = sess.get_inputs()[0].name
input_name1 = sess.get_inputs()[1].name
input_name2 = sess.get_inputs()[2].name
label_name = sess.get_outputs()[0].name

data_0 = np.array([3, 2, 1]).astype(np.float32)
data_1 = np.array([1, 4, 4]).astype(np.float32)
data_2 = np.array([2, 5, 3]).astype(np.float32)
y_actual = np.array([3, 5, 4]).astype(np.float32)

y_pred = sess.run(
        [], {input_name: data_0,
        input_name1: data_1, input_name2: data_2,})

print("The predicted output for the operation: max", y_pred)

y_pred = np.asarray(y_pred) #converting list into an array
print(y_pred.shape)

y_pred = np.squeeze(y_pred, axis=0)
print(y_pred.shape)

compare(y_pred, y_actual)