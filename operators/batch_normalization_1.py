#batchnormalisation

import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare

# Create the inputs (ValueInfoProto)
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2, 1, 3])
S = helper.make_tensor_value_info('S', TensorProto.FLOAT, [2,])
BIAS = helper.make_tensor_value_info('BIAS', TensorProto.FLOAT, [2,])
MEAN = helper.make_tensor_value_info('MEAN', TensorProto.FLOAT, [2,])
VAR = helper.make_tensor_value_info('VAR', TensorProto.FLOAT, [2,])

# Create one output (ValueInfoProto)
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2, 1, 3])

# Create a node (NodeProto)
node_def = helper.make_node(
    'BatchNormalization',
    inputs=['X', 'S', 'BIAS', 'MEAN', 'VAR'],
    outputs=['Y'],
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [X, S, BIAS, MEAN, VAR],
    [Y],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-batchnormalization1')
print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, '../onnx_generated_models/batchnormalization1.onnx')
onnx.save(model_def, new_model_path)
print('The model is saved.')


# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, '../onnx_generated_models/batchnormalization1.onnx')
onnx_model1 = onnx.load(model_path1)
print('The model is:\n{}'.format(onnx_model1))

def _batchnorm_test_mode(x, s, bias, mean, var, epsilon=1e-5):  # type: ignore
    dims_x = len(x.shape)
    dim_ones = (1,) * (dims_x - 2)
    s = s.reshape(-1, *dim_ones)
    bias = bias.reshape(-1, *dim_ones)
    mean = mean.reshape(-1, *dim_ones)
    var = var.reshape(-1, *dim_ones)
    return s * (x - mean) / np.sqrt(var + epsilon) + bias

# input size: (1, 2, 1, 3)
x = np.array([[[[-1, 0, 1]], [[2, 3, 4]]]]).astype(np.float32)
s = np.array([1.0, 1.5]).astype(np.float32)
bias = np.array([0, 1]).astype(np.float32)
mean = np.array([0, 3]).astype(np.float32)
var = np.array([1, 1.5]).astype(np.float32)
y_actual = _batchnorm_test_mode(x, s, bias, mean, var).astype(np.float32)


#Running the model using ONNX Runtime
import onnxruntime as rt
import numpy
sess = rt.InferenceSession("../onnx_generated_models/batchnormalization1.onnx")
input_name = sess.get_inputs()[0].name
input_name1 = sess.get_inputs()[1].name
input_name2 = sess.get_inputs()[2].name
input_name3 = sess.get_inputs()[3].name
input_name4 = sess.get_inputs()[4].name
label_name = sess.get_outputs()[0].name

y_pred = sess.run(
        [label_name], {input_name: x.astype(numpy.float32),
        input_name1: s.astype(numpy.float32), input_name2: bias.astype(numpy.float32),
        input_name3: mean.astype(numpy.float32), input_name4: var.astype(numpy.float32)})

print("The predicted output for the operation: BatchNormalization")
print(y_pred)

y_pred = np.asarray(y_pred) #converting list into an array
print(y_pred.shape)

y_pred = np.squeeze(y_pred, axis=0)
print(y_pred.shape)

compare(y_actual, y_pred)