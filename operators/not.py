#not (2d)
import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare

# Create one Input (ValueInfoProto)
x = helper.make_tensor_value_info('x', TensorProto.BOOL, [3, 4])

# Create one output (ValueInfoProto)
not1 = helper.make_tensor_value_info('not1', TensorProto.BOOL, [3, 4])

# Create a node (NodeProto)
node_def = helper.make_node(
    'Not',
    inputs=['x'],
    outputs=['not1'],
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [x],
    [not1],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-not')

print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, '../onnx_generated_models/onnx-not.onnx')
onnx.save(model_def, new_model_path)

print('The model is saved.')



# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, '../onnx_generated_models/onnx-not.onnx')
onnx_model1 = onnx.load(model_path1)

print('The model is:\n{}'.format(onnx_model1))

import onnxruntime as rt
sess = rt.InferenceSession("../onnx_generated_models/onnx-not.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

# 2d
x = (np.random.randn(3, 4) > 0).astype(np.bool)
y_actual = np.logical_not(x)
y_pred = sess.run(
        [], {input_name: x})

print("The predicted output for the operation: not (2d)", y_pred)

y_pred = np.asarray(y_pred) #converting list into an array
print(y_pred.shape)

y_pred = np.squeeze(y_pred, axis=0)
print(y_pred.shape)

compare(y_pred, y_actual)