#InstanceNormalization
import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare

# Create the inputs (ValueInfoProto)
x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 2, 1, 3])
s = helper.make_tensor_value_info('s', TensorProto.FLOAT, [2,])
bias = helper.make_tensor_value_info('bias', TensorProto.FLOAT, [2,])

# Create one output (ValueInfoProto)
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2, 1, 3])

# Create a node (NodeProto)
node_def = helper.make_node(
    'InstanceNormalization',
    inputs=['x', 's', 'bias'],
    outputs=['Y'],
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [x, s, bias],
    [Y],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-instanceNormalization')
print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

#model_def = onnx.utils.polish_model(model_def)
# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, '../onnx_generated_models/instanceNormalization.onnx')
onnx.save(model_def, new_model_path)
print('The model is saved.')


# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, '../onnx_generated_models/instanceNormalization.onnx')
onnx_model1 = onnx.load(model_path1)
print('The model is:\n{}'.format(onnx_model1))

def _instancenorm_test_mode(x, s, bias, epsilon=1e-5):  # type: ignore
    dims_x = len(x.shape)
    axis = tuple(range(2, dims_x))
    mean = np.mean(x, axis=axis, keepdims=True)
    var = np.var(x, axis=axis, keepdims=True)
    dim_ones = (1,) * (dims_x - 2)
    s = s.reshape(-1, *dim_ones)
    bias = bias.reshape(-1, *dim_ones)
    return s * (x - mean) / np.sqrt(var + epsilon) + bias

# input size: (1, 2, 1, 3)
x = np.array([[[[-1, 0, 1]], [[2, 3, 4]]]]).astype(np.float32)
s = np.array([1.0, 1.5]).astype(np.float32)
bias = np.array([0, 1]).astype(np.float32)
y_actual = _instancenorm_test_mode(x, s, bias).astype(np.float32)

#Running the model using ONNX Runtime
import onnxruntime as rt
import numpy
sess = rt.InferenceSession("../onnx_generated_models/instanceNormalization.onnx")
input_name = sess.get_inputs()[0].name
input_name1 = sess.get_inputs()[1].name
input_name2 = sess.get_inputs()[2].name
label_name = sess.get_outputs()[0].name


y_pred = sess.run(
        [label_name], {input_name: x.astype(numpy.float32),
        input_name1: s.astype(numpy.float32), input_name2: bias.astype(numpy.float32)})

print("The predicted output for the operation: InstanceNormalization")
print(y_pred)

y_pred = np.asarray(y_pred) #converting list into an array
print(y_pred.shape)

y_pred = np.squeeze(y_pred, axis=0)
print(y_pred.shape)

y_pred = np.round(y_pred, 5)
y_actual = np.round(y_actual, 5)

compare(y_actual, y_pred)