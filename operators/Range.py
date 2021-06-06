#Range(range_float_type_positive_delta)

import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare

# Create the inputs (ValueInfoProto)
start =  helper.make_tensor_value_info('start', TensorProto.FLOAT, [])
limit =  helper.make_tensor_value_info('limit', TensorProto.FLOAT, [])
delta =  helper.make_tensor_value_info('delta', TensorProto.FLOAT, [])


# Create one output (ValueInfoProto)
y =  helper.make_tensor_value_info('y', TensorProto.FLOAT, [2,])


# Create a node (NodeProto)
node_def = helper.make_node(
    'Range',
    inputs=['start', 'limit', 'delta'],
    outputs=['y'],
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [start, limit, delta],
    [y],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-Range')
print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, '../onnx_generated_models/Range.onnx')
onnx.save(model_def, new_model_path)
print('The model is saved.')


# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, '../onnx_generated_models/Range.onnx')
onnx_model1 = onnx.load(model_path1)
print('The model is:\n{}'.format(onnx_model1))

start = np.array([1], dtype=np.float32)
limit = np.array([5], dtype=np.float32)
delta = np.array([2], dtype=np.float32)

y_actual = np.arange(start, limit, delta, dtype=np.float32)  # expected output [1.0, 3.0]

#Running the model using ONNX Runtime
import onnxruntime as rt
import numpy
sess = rt.InferenceSession("../onnx_generated_models/Range.onnx")
input_name = sess.get_inputs()[0].name
input_name1 = sess.get_inputs()[1].name
input_name2 = sess.get_inputs()[2].name
label_name = sess.get_outputs()[0].name

y_pred = sess.run(
        [label_name], {input_name: start,
        input_name1: limit, input_name2: delta,
        })

print("The predicted output for the operation: Range")
print(y_pred)

y_pred = np.asarray(y_pred) #converting list into an array
print(y_pred.shape)

y_pred = np.squeeze(y_pred, axis=0)
print(y_pred.shape)

compare(y_actual, y_pred)