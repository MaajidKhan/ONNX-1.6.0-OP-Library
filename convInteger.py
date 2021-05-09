## ConvInteger without padding

import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare

# Create the inputs (ValueInfoProto)
X = helper.make_tensor_value_info('X', TensorProto.UINT8, [1, 1, 3, 3])
w = helper.make_tensor_value_info('w', TensorProto.UINT8, [1, 1, 2, 2])
# Create one output (ValueInfoProto)
Y = helper.make_tensor_value_info('Y', TensorProto.INT32, [1, 1, 2, 2])

# Create a node (NodeProto)
node_def = helper.make_node(
    'ConvInteger',
    inputs=['X', 'w'],
    outputs=['Y'],
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [X, w],
    [Y],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-ConvInteger_1')
print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

#model_def = onnx.utils.polish_model(model_def)
# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, 'ConvInteger_1.onnx')
onnx.save(model_def, new_model_path)
print('The model is saved.')


# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, 'ConvInteger_1.onnx')
onnx_model1 = onnx.load(model_path1)
print('The model is:\n{}'.format(onnx_model1))

x = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.uint8).reshape((1, 1, 3, 3))
x_zero_point = np.uint8(1)
w = np.array([1, 1, 1, 1]).astype(np.uint8).reshape((1, 1, 2, 2))

y_actual = np.array([16, 20, 28, 32]).astype(np.int32).reshape(1, 1, 2, 2)


#Running the model using ONNX Runtime
import onnxruntime as rt
import numpy
sess = rt.InferenceSession("ConvInteger_1.onnx")
input_name = sess.get_inputs()[0].name
input_name1 = sess.get_inputs()[1].name
label_name = sess.get_outputs()[0].name


y_pred = sess.run(
        [label_name], {input_name: x,
        input_name1: w})

print("The predicted output for the operation: ConvInteger")
print(y_pred)

y_pred = np.asarray(y_pred) #converting list into an array
print(y_pred.shape)

y_pred = np.squeeze(y_pred, axis=0)
print(y_pred.shape)

compare(y_actual, y_pred)