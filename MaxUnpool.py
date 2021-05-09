#MaxUnpool (with_output_shape)
import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare

# Create one Input (ValueInfoProto)
xT = helper.make_tensor_value_info('xT', TensorProto.FLOAT, [1, 1, 2, 2])
xI = helper.make_tensor_value_info('xI', TensorProto.INT64, [1, 1, 2, 2])
output_shape = helper.make_tensor_value_info('output_shape', TensorProto.INT64, [4,])


# Create one output (ValueInfoProto)
y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 5, 5])

# Create a node (NodeProto)
node_def = helper.make_node(
    'MaxUnpool',
    inputs=['xT', 'xI', 'output_shape'],
    outputs=['y'],
    kernel_shape=[2, 2],
    strides=[2, 2]
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [xT, xI, output_shape],
    [y],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-MaxUnpool')

print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, 'onnx-MaxUnpool.onnx')
onnx.save(model_def, new_model_path)

print('The model is saved.')



# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, 'onnx-MaxUnpool.onnx')
onnx_model1 = onnx.load(model_path1)

print('The model is:\n{}'.format(onnx_model1))

import onnxruntime as rt
sess = rt.InferenceSession("onnx-MaxUnpool.onnx")
input_name = sess.get_inputs()[0].name
input_name1 = sess.get_inputs()[1].name
input_name2 = sess.get_inputs()[2].name
label_name = sess.get_outputs()[0].name

xT = np.array([[[[5, 6],
                 [7, 8]]]], dtype=np.float32)
xI = np.array([[[[5, 7],
                 [13, 15]]]], dtype=np.int64)
output_shape = np.array((1, 1, 5, 5), dtype=np.int64)
y_actual = np.array([[[[0, 0, 0, 0, 0],
                [0, 5, 0, 6, 0],
                [0, 0, 0, 0, 0],
                [0, 7, 0, 8, 0],
                [0, 0, 0, 0, 0]]]], dtype=np.float32)

y_pred = sess.run(
        [], {input_name: xT,
        input_name1: xI, input_name2: output_shape})

print("The predicted output for the operation: MaxUnpool (with_output_shape)", y_pred)

y_pred = np.asarray(y_pred) #converting list into an array
print(y_pred.shape)

y_pred = np.squeeze(y_pred, axis=0)
print(y_pred.shape)

y_pred1 = np.round(y_pred, 6)
y_actual1 = np.round(y_actual, 6)
compare(y_pred1, y_actual1)