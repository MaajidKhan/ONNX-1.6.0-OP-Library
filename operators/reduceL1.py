#reduceL1 (default_axes_keepdims)
import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare

# The protobuf definition can be found here:
# https://github.com/onnx/onnx/blob/master/onnx/onnx.proto

X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [3, 2, 2])

# Create one output (ValueInfoProto)
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 1, 1])

keepdims = 1
# Create a node (NodeProto)
node_def = helper.make_node(
    'ReduceL1',
    inputs=['X'],
    outputs=['Y'],
    keepdims=keepdims
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [X],
    [Y],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-reduceL1')

print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, '../onnx_generated_models/onnx-reduceL1.onnx')
onnx.save(model_def, new_model_path)

print('The model is saved.')



# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, '../onnx_generated_models/onnx-reduceL1.onnx')
onnx_model1 = onnx.load(model_path1)

print('The model is:\n{}'.format(onnx_model1))



import onnxruntime as rt
import numpy as np
sess = rt.InferenceSession("../onnx_generated_models/onnx-reduceL1.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

shape = [3, 2, 2]
axes = None
data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)


y_pred = sess.run(
        [], {input_name: data})

y_actual = np.sum(a=np.abs(data), axis=axes, keepdims=keepdims == 1)
#print(reduced)
print("The input is:")
print(data)
print("\n")

print("The predicted output for ReduceL1 (default_axes_keepdims) is:")
print(y_pred)

y_pred1 = y_pred[0][0]
y_actual1 = y_actual[0]

compare(y_pred1, y_actual1)