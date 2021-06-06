#OneHot (with_axis)
import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare

# Create one Input (ValueInfoProto)
indices = helper.make_tensor_value_info('indices', TensorProto.FLOAT, [2, 2])
depth = helper.make_tensor_value_info('depth', TensorProto.FLOAT, [1])
values = helper.make_tensor_value_info('values', TensorProto.FLOAT, [2])

# Create one output (ValueInfoProto)
y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 10, 2])

axisValue = 1
on_value = 3
off_value = 1
output_type = np.float32

# Create a node (NodeProto)
node_def = helper.make_node(
    'OneHot',
    inputs=['indices', 'depth', 'values'],
    outputs=['y'],
    axis=axisValue
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [indices, depth, values],
    [y],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-OneHot')

print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, '../onnx_generated_models/onnx-OneHot.onnx')
onnx.save(model_def, new_model_path)

print('The model is saved.')



# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, '../onnx_generated_models/onnx-OneHot.onnx')
onnx_model1 = onnx.load(model_path1)

print('The model is:\n{}'.format(onnx_model1))

import onnxruntime as rt
sess = rt.InferenceSession("../onnx_generated_models/onnx-OneHot.onnx")
input_name = sess.get_inputs()[0].name
input_name1 = sess.get_inputs()[1].name
input_name2 = sess.get_inputs()[2].name
label_name = sess.get_outputs()[0].name

def one_hot(indices, depth, axis=-1, dtype=np.float32):  # type: ignore
    ''' Compute one hot from indices at a specific axis '''
    values = np.asarray(indices)
    rank = len(values.shape)
    depth_range = np.arange(depth)
    if axis < 0:
        axis += (rank + 1)
    ls = values.shape[0:axis]
    rs = values.shape[axis:rank]
    targets = np.reshape(depth_range, (1,) * len(ls) + depth_range.shape + (1,) * len(rs))
    values = np.reshape(np.mod(values, depth), ls + (1,) + rs)
    return np.asarray(targets == values, dtype=dtype)

indices = np.array([[1, 9],
                    [2, 4]], dtype=np.float32)
depth = np.array([10], dtype=np.float32)
values = np.array([off_value, on_value], dtype=output_type)
y = one_hot(indices, depth, axis=axisValue, dtype=output_type)
y_actual = y * (on_value - off_value) + off_value

y_pred = sess.run(
        [], {input_name: indices,
        input_name1: depth, input_name2: values})

print("The predicted output for the operation: OneHot (with_axis)", y_pred)

y_pred = np.asarray(y_pred) #converting list into an array
print(y_pred.shape)

y_pred = np.squeeze(y_pred, axis=0)
print(y_pred.shape)

compare(y_pred, y_actual)