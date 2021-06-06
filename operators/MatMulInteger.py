#MatMulInteger

import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare

# Create one Input (ValueInfoProto)
A = helper.make_tensor_value_info('A', TensorProto.UINT8, [4, 3])
B = helper.make_tensor_value_info('B', TensorProto.UINT8, [3, 2])
a_zero_point = helper.make_tensor_value_info('a_zero_point', TensorProto.UINT8, [1])
b_zero_point = helper.make_tensor_value_info('b_zero_point', TensorProto.UINT8, [1])

# Create one output (ValueInfoProto)
Y = helper.make_tensor_value_info('Y', TensorProto.INT32, [4, 2])

# Create a node (NodeProto)
node_def = helper.make_node(
    'MatMulInteger',
    inputs=['A', 'B', 'a_zero_point', 'b_zero_point'],
    outputs=['Y'],
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [A, B, a_zero_point, b_zero_point],
    [Y],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-MatMulInteger')

print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, '../onnx_generated_models/onnx-MatMulInteger.onnx')
onnx.save(model_def, new_model_path)

print('The model is saved.')



# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, '../onnx_generated_models/onnx-MatMulInteger.onnx')
onnx_model1 = onnx.load(model_path1)

print('The model is:\n{}'.format(onnx_model1))

import onnxruntime as rt
sess = rt.InferenceSession("../onnx_generated_models/onnx-MatMulInteger.onnx")
input_name = sess.get_inputs()[0].name
input_name1 = sess.get_inputs()[1].name
input_name2 = sess.get_inputs()[2].name
input_name3 = sess.get_inputs()[3].name
label_name = sess.get_outputs()[0].name


A = np.array([[11, 7, 3],
    [10, 6, 2],
    [9, 5, 1],
    [8, 4, 0], ], dtype=np.uint8)

a_zero_point = np.array([12], dtype=np.uint8)

B = np.array([[1, 4],
    [2, 5],
    [3, 6], ], dtype=np.uint8)

b_zero_point = np.array([0], dtype=np.uint8)

y_actual = np.array([[-38, -83],
    [-44, -98],
    [-50, -113],
    [-56, -128], ], dtype=np.int32)

y_pred = sess.run(
        [label_name], {input_name: A,
        input_name1: B , input_name2: a_zero_point,
        input_name3: b_zero_point
       })

print("The predicted output for the operation: MatMulInteger (2d)", y_pred)

y_pred = np.asarray(y_pred) #converting list into an array
print(y_pred.shape)

y_pred = np.squeeze(y_pred, axis=0)
print(y_pred.shape)


compare(y_pred, y_actual)