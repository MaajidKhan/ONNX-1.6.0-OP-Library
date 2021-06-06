# resize (resize_downsample_scales_cubic)
import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare

# The protobuf definition can be found here:
# https://github.com/onnx/onnx/blob/master/onnx/onnx.proto

X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 1, 4, 4])

roi= helper.make_tensor_value_info('roi', TensorProto.FLOAT, [0,])

scales = helper.make_tensor_value_info('scales', TensorProto.FLOAT, [4,])

# Create one output (ValueInfoProto)
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 1, 3, 3])


# Create a node (NodeProto)
node_def = helper.make_node(
    'Resize',
    inputs=['X', 'roi', 'scales'],
    outputs=['Y'],
    mode='cubic',
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [X, roi, scales],
    [Y],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-resize')

print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, '../onnx_generated_models/onnx-resize.onnx')
onnx.save(model_def, new_model_path)

print('The model is saved.')



# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, '../onnx_generated_models/onnx-resize.onnx')
onnx_model1 = onnx.load(model_path1)

print('The model is:\n{}'.format(onnx_model1))



import onnxruntime as rt
import numpy as np
sess = rt.InferenceSession("../onnx_generated_models/onnx-resize.onnx")
input_name = sess.get_inputs()[0].name
input_name1 = sess.get_inputs()[1].name
input_name2 = sess.get_inputs()[2].name
label_name = sess.get_outputs()[0].name

data = np.array([[[
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
]]], dtype=np.float32)

roi = np.array([], dtype=np.float32)

scales = np.array([1.0, 1.0, 0.8, 0.8], dtype=np.float32)

y_actual = np.array([[[[1.47119141,2.78125,4.08251953],
[ 6.71142578 , 8.02148438 , 9.32275391],
[11.91650391 ,13.2265625,  14.52783203]]]], dtype=np.float32)



y_pred = sess.run(
        [], {input_name: data,
        input_name1: roi, input_name2: scales })

print(y_pred)
print("The predicted output for the operation: resize (resize_downsample_scales_cubic)", y_pred)

y_pred = np.asarray(y_pred) #converting list into an array
print(y_pred.shape)

y_pred = np.squeeze(y_pred, axis=0)
print(y_pred.shape)

y_pred1 = np.round(y_pred, 5)
y_actual1 = np.round(y_actual, 5)
compare(y_pred1, y_actual1)