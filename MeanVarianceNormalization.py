#MeanVarianceNormalization
import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare

# Create one Input (ValueInfoProto)
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [3, 3, 3, 1])


# Create one output (ValueInfoProto)
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [3, 3, 3, 1])

# Create a node (NodeProto)
node_def = helper.make_node(
    'MeanVarianceNormalization',
    inputs=['X'],
    outputs=['Y']
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [X],
    [Y],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-MeanVarianceNormalization')

print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, 'onnx-MeanVarianceNormalization.onnx')
onnx.save(model_def, new_model_path)

print('The model is saved.')



# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, 'onnx-MeanVarianceNormalization.onnx')
onnx_model1 = onnx.load(model_path1)

print('The model is:\n{}'.format(onnx_model1))

import onnxruntime as rt
sess = rt.InferenceSession("onnx-MeanVarianceNormalization.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

input_data = np.array([[[[0.8439683], [0.5665144], [0.05836735]],
    [[0.02916367], [0.12964272], [0.5060197]],
    [[0.79538304], [0.9411346], [0.9546573]]],
    [[[0.17730942], [0.46192095], [0.26480448]],
    [[0.6746842], [0.01665257], [0.62473077]],
    [[0.9240844], [0.9722341], [0.11965699]]],
    [[[0.41356155], [0.9129373], [0.59330076]],
    [[0.81929934], [0.7862604], [0.11799799]],
    [[0.69248444], [0.54119414], [0.07513223]]]], dtype=np.float32)

# Calculate expected output data
data_mean = np.mean(input_data, axis=(0, 2, 3), keepdims=1)
data_mean_squared = np.power(data_mean, 2)
data_squared = np.power(input_data, 2)
data_squared_mean = np.mean(data_squared, axis=(0, 2, 3), keepdims=1)
std = np.sqrt(data_squared_mean - data_mean_squared)
y_actual = (input_data - data_mean) / (std + 1e-9)

y_pred = sess.run(
        [], {input_name: input_data})

print("The predicted output for the operation: MeanVarianceNormalization", y_pred)

y_pred = np.asarray(y_pred) #converting list into an array
print(y_pred.shape)

y_pred = np.squeeze(y_pred, axis=0)
print(y_pred.shape)

y_pred1 = np.round(y_pred, 5)
y_actual1 = np.round(y_actual, 5)
compare(y_pred1, y_actual1)