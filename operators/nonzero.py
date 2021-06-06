#nonzero
import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare

# Create one Input (ValueInfoProto)
condition = helper.make_tensor_value_info('condition', TensorProto.BOOL, [2, 2])

# Create one output (ValueInfoProto)
result = helper.make_tensor_value_info('result', TensorProto.INT64, [2, 3])

# Create a node (NodeProto)
node_def = helper.make_node(
    'NonZero',
    inputs=['condition'],
    outputs=['result'],
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [condition],
    [result],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-nonzero')

print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, '../onnx_generated_models/onnx-nonzero.onnx')
onnx.save(model_def, new_model_path)

print('The model is saved.')



# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, '../onnx_generated_models/onnx-nonzero.onnx')
onnx_model1 = onnx.load(model_path1)

print('The model is:\n{}'.format(onnx_model1))

import onnxruntime as rt
sess = rt.InferenceSession("../onnx_generated_models/onnx-nonzero.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name


condition = np.array([[1, 0], [1, 1]], dtype=np.bool)
y_actual = np.array((np.nonzero(condition)))  # expected output [[0, 1, 1], [0, 0, 1]]

y_pred = sess.run(
        [], {input_name: condition})

print("The predicted output for the operation: nonzero", y_pred)

y_pred = np.asarray(y_pred) #converting list into an array
print(y_pred.shape)

y_pred = np.squeeze(y_pred, axis=0)
print(y_pred.shape)

compare(y_pred, y_actual)