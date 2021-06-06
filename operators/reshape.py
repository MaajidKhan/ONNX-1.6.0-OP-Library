# reshape
import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare

# The protobuf definition can be found here:
# https://github.com/onnx/onnx/blob/master/onnx/onnx.proto

X = helper.make_tensor_value_info('X', TensorProto.INT64, [2, 3, 4])

shape= helper.make_tensor_value_info('shape', TensorProto.INT64, [3,])

# Create one output (ValueInfoProto)
Y = helper.make_tensor_value_info('Y', TensorProto.INT64, [4, 2, 3])


# Create a node (NodeProto)
node_def = helper.make_node(
    'Reshape', # node name
    ['X','shape'], # inputs
    ['Y'], # outputs
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [X, shape],
    [Y],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-reshape')

print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, '../onnx_generated_models/onnx-reshape.onnx')
onnx.save(model_def, new_model_path)

print('The model is saved.')



# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, '../onnx_generated_models/onnx-reshape.onnx')
onnx_model1 = onnx.load(model_path1)

print('The model is:\n{}'.format(onnx_model1))



import onnxruntime as rt
import numpy as np
sess = rt.InferenceSession("../onnx_generated_models/onnx-reshape.onnx")
input_name = sess.get_inputs()[0].name
input_name1 = sess.get_inputs()[1].name
label_name = sess.get_outputs()[0].name

original_shape = [2, 3, 4]
data = np.random.random_sample(original_shape).astype(np.int64)

rs = np.array([4, 2, 3], dtype=np.int64)

y_actual = data.reshape(4, 2, 3)
y_pred = sess.run(
        [], {input_name: data,
        input_name1: rs })

print("The predicted output for the operation: reshape:", y_pred)

y_pred = np.asarray(y_pred) #converting list into an array
print(y_pred.shape)

y_pred = np.squeeze(y_pred, axis=0)
print(y_pred.shape)

compare(y_pred, y_actual)