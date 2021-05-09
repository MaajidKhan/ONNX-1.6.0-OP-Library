#GatherND (float32)
import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare

# Create one Input (ValueInfoProto)
data = helper.make_tensor_value_info('data', TensorProto.FLOAT, [2, 2, 2])
indices = helper.make_tensor_value_info('indices', TensorProto.INT64, [2, 1, 2])

# Create one output (ValueInfoProto)
output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 1, 2])


# Create a node (NodeProto)
node_def = helper.make_node(
    'GatherND',
    inputs=['data', 'indices'],
    outputs=['output'],
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [data, indices],
    [output],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-GatherND')

print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, 'onnx-GatherND.onnx')
onnx.save(model_def, new_model_path)

print('The model is saved.')



# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, 'onnx-GatherND.onnx')
onnx_model1 = onnx.load(model_path1)

print('The model is:\n{}'.format(onnx_model1))

import onnxruntime as rt
sess = rt.InferenceSession("onnx-GatherND.onnx")
input_name = sess.get_inputs()[0].name
input_name1 = sess.get_inputs()[1].name
label_name = sess.get_outputs()[0].name

def gather_nd_impl(data, indices):
    # type: (np.ndarray, np.ndarray) -> np.ndarray

    # Note the data rank - will be reused multiple times later
    data_rank = len(data.shape)

    # Check input tensors' shape/rank condition
    assert indices.shape[-1] <= data_rank

    # Compute output of the op as below
    # Compute shape of output array
    output_shape = list(indices.shape)[:-1] if (indices.shape[-1] == data_rank) else list(indices.shape)[:-1] + list(data.shape)[indices.shape[-1]:]

    # Placeholder for output data
    output_data_buffer = []

    # Flatten 'indices' to 2D array
    reshaped_indices = indices.reshape(-1, indices.shape[-1])

    # gather each scalar value from 'data'
    for outer_dim in range(reshaped_indices.shape[0]):
        gather_index = tuple(reshaped_indices[outer_dim])
        output_data_buffer.append(data[gather_index])
    return np.asarray(output_data_buffer, dtype=data.dtype).reshape(output_shape)


data = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.float32)
indices = np.array([[[0, 1]], [[1, 0]]], dtype=np.int64)
output = gather_nd_impl(data, indices)
y_actual = np.array([[[2, 3]], [[4, 5]]], dtype=np.float32)

y_pred = sess.run(
        [], {input_name: data,
        input_name1: indices})

print("The predicted output for the operation: GatherND (float32)", y_pred)

y_pred = np.asarray(y_pred) #converting list into an array
print(y_pred.shape)

y_pred = np.squeeze(y_pred, axis=0)
print(y_pred.shape)

compare(y_pred, y_actual)

