#MatMul
import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare

# Create one Input (ValueInfoProto)
a = helper.make_tensor_value_info('a', TensorProto.FLOAT, [3, 4])
b = helper.make_tensor_value_info('b', TensorProto.FLOAT, [4, 3])

# Create one output (ValueInfoProto)
c = helper.make_tensor_value_info('c', TensorProto.FLOAT, [3, 3])

# Create a node (NodeProto)
node_def = helper.make_node(
    'MatMul',
    inputs=['a', 'b'],
    outputs=['c'],
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [a, b],
    [c],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-MatMul')

print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, 'onnx-MatMul.onnx')
onnx.save(model_def, new_model_path)

print('The model is saved.')



# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, 'onnx-MatMul.onnx')
onnx_model1 = onnx.load(model_path1)

print('The model is:\n{}'.format(onnx_model1))

import onnxruntime as rt
sess = rt.InferenceSession("onnx-MatMul.onnx")
input_name = sess.get_inputs()[0].name
input_name1 = sess.get_inputs()[1].name
label_name = sess.get_outputs()[0].name

# 2d
a = np.random.randn(3, 4).astype(np.float32)
b = np.random.randn(4, 3).astype(np.float32)
y_actual = np.matmul(a, b)

y_pred = sess.run(
        [], {input_name: a,
        input_name1: b,
       })

print("The predicted output for the operation: MatMul", y_pred)

y_pred = np.asarray(y_pred) #converting list into an array
print(y_pred.shape)

y_pred = np.squeeze(y_pred, axis=0)
print(y_pred.shape)

y_pred1 = np.round(y_pred, 6)
y_actual1 = np.round(y_actual, 6)
compare(y_pred1, y_actual1)