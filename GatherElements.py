#GatherElements (gather_elements_0)
import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare

# Create one Input (ValueInfoProto)
data = helper.make_tensor_value_info('data', TensorProto.FLOAT, [2, 2])
indices = helper.make_tensor_value_info('indices', TensorProto.INT32, [2, 2])

# Create one output (ValueInfoProto)
y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 2])

axis = 1
# Create a node (NodeProto)
node_def = helper.make_node(
    'GatherElements',
    inputs=['data', 'indices'],
    outputs=['y'],
    axis=axis,
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [data, indices],
    [y],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-GatherElements')

print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, 'onnx-GatherElements.onnx')
onnx.save(model_def, new_model_path)

print('The model is saved.')



# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, 'onnx-GatherElements.onnx')
onnx_model1 = onnx.load(model_path1)

print('The model is:\n{}'.format(onnx_model1))

import onnxruntime as rt
sess = rt.InferenceSession("onnx-GatherElements.onnx")
input_name = sess.get_inputs()[0].name
input_name1 = sess.get_inputs()[1].name
label_name = sess.get_outputs()[0].name

data = np.array([[1, 2],
                 [3, 4]], dtype=np.float32)
indices = np.array([[0, 0],
                    [1, 0]], dtype=np.int32)

# The below GatherElements' numpy implementation is from https://stackoverflow.com/a/46204790/11767360
def gather_elements(data, indices, axis=0):  # type: ignore
    data_swaped = np.swapaxes(data, 0, axis)
    index_swaped = np.swapaxes(indices, 0, axis)
    gathered = np.choose(index_swaped, data_swaped, mode='wrap')
    y = np.swapaxes(gathered, 0, axis)
    return y

y_actual = gather_elements(data, indices, axis)
# print(y) produces
# [[1, 1],
#  [4, 3]]

y_pred = sess.run(
        [], {input_name: data,
        input_name1: indices})

print("The predicted output for the operation: GatherElements (gather_elements_0)", y_pred)

y_pred = np.asarray(y_pred) #converting list into an array
print(y_pred.shape)

y_pred = np.squeeze(y_pred, axis=0)
print(y_pred.shape)

compare(y_pred, y_actual)

