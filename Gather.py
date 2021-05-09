#Gather (gather_0)
import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare

# Create one Input (ValueInfoProto)
data = helper.make_tensor_value_info('data', TensorProto.FLOAT, [5, 4, 3, 2])
indices = helper.make_tensor_value_info('indices', TensorProto.INT32, [3])

# Create one output (ValueInfoProto)
y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 4, 3, 2])

# Create a node (NodeProto)
node_def = helper.make_node(
    'Gather',
    inputs=['data', 'indices'],
    outputs=['y'],
    axis=0,
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [data, indices],
    [y],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-Gather')

print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, 'onnx-Gather.onnx')
onnx.save(model_def, new_model_path)

print('The model is saved.')



# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, 'onnx-Gather.onnx')
onnx_model1 = onnx.load(model_path1)

print('The model is:\n{}'.format(onnx_model1))

import onnxruntime as rt
sess = rt.InferenceSession("onnx-Gather.onnx")
input_name = sess.get_inputs()[0].name
input_name1 = sess.get_inputs()[1].name
label_name = sess.get_outputs()[0].name

data = np.random.randn(5, 4, 3, 2).astype(np.float32)
indices = np.array([0, 1, 3])
y_actual = np.take(data, indices, axis=0)

y_pred = sess.run(
        [], {input_name: data,
        input_name1: indices})

print("The predicted output for the operation: Gather (gather_0)", y_pred)

y_pred = np.asarray(y_pred) #converting list into an array
print(y_pred.shape)

y_pred = np.squeeze(y_pred, axis=0)
print(y_pred.shape)

compare(y_pred, y_actual)

