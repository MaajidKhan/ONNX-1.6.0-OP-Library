#Compress(compress_0)

import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare

# Create the inputs (ValueInfoProto)
input =  helper.make_tensor_value_info('input', TensorProto.FLOAT, [3, 2])
condition =  helper.make_tensor_value_info('condition', TensorProto.BOOL, [3,])


# Create one output (ValueInfoProto)
output =  helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 2])


# Create a node (NodeProto)
node_def = helper.make_node(
    'Compress',
    inputs=['input', 'condition'],
    outputs=['output'],
    axis=0,
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [input, condition],
    [output],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-Compress')
print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, '../onnx_generated_models/Compress.onnx')
onnx.save(model_def, new_model_path)
print('The model is saved.')


# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, '../onnx_generated_models/Compress.onnx')
onnx_model1 = onnx.load(model_path1)
print('The model is:\n{}'.format(onnx_model1))

input = np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float32)
condition = np.array([0, 1, 1]).astype(np.bool)
y_actual = np.compress(condition, input, axis=0)
#print(output)
#[[ 3.  4.]
# [ 5.  6.]]


#Running the model using ONNX Runtime
import onnxruntime as rt
import numpy
sess = rt.InferenceSession("../onnx_generated_models/Compress.onnx")
input_name = sess.get_inputs()[0].name
input_name1 = sess.get_inputs()[1].name
label_name = sess.get_outputs()[0].name

y_pred = sess.run(
        [label_name], {input_name: input,
        input_name1: condition
        })

print("The predicted output for the operation: Compress")
print(y_pred)

y_pred = np.asarray(y_pred) #converting list into an array
print(y_pred.shape)

y_pred = np.squeeze(y_pred, axis=0)
print(y_pred.shape)

compare(y_actual, y_pred)