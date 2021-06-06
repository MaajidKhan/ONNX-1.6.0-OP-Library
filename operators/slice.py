#slice

import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare

# Create the inputs (ValueInfoProto)
x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [20, 10, 5])
starts = helper.make_tensor_value_info('starts', TensorProto.INT64, [2,])
ends = helper.make_tensor_value_info('ends', TensorProto.INT64, [2,])
axes = helper.make_tensor_value_info('axes', TensorProto.INT64, [2,])
steps = helper.make_tensor_value_info('steps', TensorProto.INT64, [2,])


# Create one output (ValueInfoProto)
y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3, 10, 5])

# Create a node (NodeProto)
node_def = helper.make_node(
    'Slice',
    inputs=['x', 'starts', 'ends', 'axes', 'steps'],
    outputs=['y'],
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [x, starts, ends, axes, steps],
    [y],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-slice')
print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, '../onnx_generated_models/slice.onnx')
onnx.save(model_def, new_model_path)
print('The model is saved.')


# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, '../onnx_generated_models/slice.onnx')
onnx_model1 = onnx.load(model_path1)
print('The model is:\n{}'.format(onnx_model1))


x = np.random.randn(20, 10, 5).astype(np.float32)
y_actual = x[0:3, 0:10]
starts = np.array([0, 0], dtype=np.int64)
ends = np.array([3, 10], dtype=np.int64)
axes = np.array([0, 1], dtype=np.int64)
steps = np.array([1, 1], dtype=np.int64)

#Running the model using ONNX Runtime
import onnxruntime as rt
import numpy
sess = rt.InferenceSession("../onnx_generated_models/slice.onnx")
input_name = sess.get_inputs()[0].name
input_name1 = sess.get_inputs()[1].name
input_name2 = sess.get_inputs()[2].name
input_name3 = sess.get_inputs()[3].name
input_name4 = sess.get_inputs()[4].name
label_name = sess.get_outputs()[0].name

y_pred = sess.run(
        [label_name], {input_name: x.astype(numpy.float32),
        input_name1: starts.astype(numpy.int64), input_name2: ends.astype(numpy.int64),
        input_name3: axes.astype(numpy.int64), input_name4: steps.astype(numpy.int64),
        })

print("The predicted output for the operation: slice")
print(y_pred)

y_pred = np.asarray(y_pred) #converting list into an array
print(y_pred.shape)

y_pred = np.squeeze(y_pred, axis=0)
print(y_pred.shape)

compare(y_actual, y_pred)