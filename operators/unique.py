#unique (not_sorted_without_axis)

import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare

# Create the inputs (ValueInfoProto)
x =  helper.make_tensor_value_info('x', TensorProto.FLOAT, [6,])

# Create one output (ValueInfoProto)
y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [4])
indices = helper.make_tensor_value_info('indices', TensorProto.INT64, [4])
inverse_indices = helper.make_tensor_value_info('inverse_indices', TensorProto.INT64, [6])
counts = helper.make_tensor_value_info('counts', TensorProto.INT64, [4])



default_alpha = 1.0
# Create a node (NodeProto)
node_def = helper.make_node(
    'Unique',
    inputs=['x'],
    outputs=['y', 'indices', 'inverse_indices', 'counts'],
    sorted=0
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [x],
    [y, indices, inverse_indices, counts],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-unique')
print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, '../onnx_generated_models/unique.onnx')
onnx.save(model_def, new_model_path)
print('The model is saved.')


# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, '../onnx_generated_models/unique.onnx')
onnx_model1 = onnx.load(model_path1)
print('The model is:\n{}'.format(onnx_model1))

x = np.array([2.0, 1.0, 1.0, 3.0, 4.0, 3.0], dtype=np.float32)
y_actual, indices, inverse_indices, counts = np.unique(x, True, True, True)

# prepare index mapping from sorted to unsorted
argsorted_indices = np.argsort(indices)
inverse_indices_map = {i: si for i, si in zip(argsorted_indices, np.arange(len(argsorted_indices)))}

y_actual = np.take(x, indices, axis = 0)
indices = indices[argsorted_indices]
inverse_indices = np.asarray([inverse_indices_map[i] for i in inverse_indices], dtype=np.int64)
counts = counts[argsorted_indices]

actual_output = []
actual_output.append(y_actual)
actual_output.append(indices)
actual_output.append(inverse_indices)
actual_output.append(counts)
# print(y)
# [2.0, 1.0, 3.0, 4.0]
# print(indices)
# [0 1 3 4]
# print(inverse_indices)
# [0, 1, 1, 2, 3, 2]
# print(counts)
# [1, 2, 2, 1]


#Running the model using ONNX Runtime
import onnxruntime as rt
import numpy
sess = rt.InferenceSession("../onnx_generated_models/unique.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
label_name1 = sess.get_outputs()[1].name
label_name2 = sess.get_outputs()[2].name
label_name3 = sess.get_outputs()[3].name

y_pred = sess.run(
        [label_name,label_name1, label_name2, label_name3], {input_name: x.astype(numpy.float32)
        })

print("The predicted output for the operation: unique")
print(y_pred)

y_pred = np.asarray(y_pred) #converting list into an array
print(y_pred.shape)

compare(actual_output, y_pred)