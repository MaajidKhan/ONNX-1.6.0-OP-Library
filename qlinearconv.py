#qlinearconv

import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare

# Create the inputs (ValueInfoProto)
x =  helper.make_tensor_value_info('x', TensorProto.UINT8, [1, 1, 7, 7])
x_scale =  helper.make_tensor_value_info('x_scale', TensorProto.FLOAT, [1,])
x_zero_point =  helper.make_tensor_value_info('x_zero_point', TensorProto.UINT8, [1,])
w =  helper.make_tensor_value_info('w', TensorProto.UINT8, [1, 1, 1, 1])
w_scale =  helper.make_tensor_value_info('w_scale', TensorProto.FLOAT, [1,])
w_zero_point =  helper.make_tensor_value_info('w_zero_point', TensorProto.UINT8, [1,])
y_scale =  helper.make_tensor_value_info('y_scale', TensorProto.FLOAT, [1,])
y_zero_point =  helper.make_tensor_value_info('y_zero_point', TensorProto.UINT8, [1,])




# Create one output (ValueInfoProto)
output =  helper.make_tensor_value_info('output', TensorProto.UINT8, [1, 1, 7, 7])


# Create a node (NodeProto)
node_def = helper.make_node(
    'QLinearConv',
    inputs=['x', 'x_scale', 'x_zero_point', 'w', 'w_scale', 'w_zero_point', 'y_scale', 'y_zero_point'],
    outputs=['output'],
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point],
    [output],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-qlinearconv')
print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, 'qlinearconv.onnx')
onnx.save(model_def, new_model_path)
print('The model is saved.')


# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, 'qlinearconv.onnx')
onnx_model1 = onnx.load(model_path1)
print('The model is:\n{}'.format(onnx_model1))


x = np.array([[255, 174, 162, 25, 203, 168, 58],
    [15, 59, 237, 95, 129, 0, 64],
    [56, 242, 153, 221, 168, 12, 166],
    [232, 178, 186, 195, 237, 162, 237],
    [188, 39, 124, 77, 80, 102, 43],
    [127, 230, 21, 83, 41, 40, 134],
    [255, 154, 92, 141, 42, 148, 247], ], dtype=np.uint8).reshape((1, 1, 7, 7))

x_scale = np.array([0.00369204697]).astype(np.float32)
x_zero_point = np.array([132], dtype=np.uint8)

w = np.array([0], dtype=np.uint8).reshape((1, 1, 1, 1))

w_scale = np.array([0.00172794575]).astype(np.float32)
w_zero_point = np.array([255], dtype=np.uint8)

y_scale = np.array([0.00162681262]).astype(np.float32)
y_zero_point = np.array([123], dtype=np.uint8)

y_actual = np.array([[0, 81, 93, 230, 52, 87, 197],
    [240, 196, 18, 160, 126, 255, 191],
    [199, 13, 102, 34, 87, 243, 89],
    [23, 77, 69, 60, 18, 93, 18],
    [67, 216, 131, 178, 175, 153, 212],
    [128, 25, 234, 172, 214, 215, 121],
    [0, 101, 163, 114, 213, 107, 8], ], dtype=np.uint8).reshape((1, 1, 7, 7))


#Running the model using ONNX Runtime
import onnxruntime as rt
import numpy
sess = rt.InferenceSession("qlinearconv.onnx")
input_name = sess.get_inputs()[0].name
input_name1 = sess.get_inputs()[1].name
input_name2 = sess.get_inputs()[2].name
input_name3 = sess.get_inputs()[3].name
input_name4 = sess.get_inputs()[4].name
input_name5 = sess.get_inputs()[5].name
input_name6 = sess.get_inputs()[6].name
input_name7 = sess.get_inputs()[7].name
label_name = sess.get_outputs()[0].name

y_pred = sess.run(
        [label_name], {input_name: x,
        input_name1: x_scale, input_name2: x_zero_point,
        input_name3: w, input_name4: w_scale,
        input_name5: w_zero_point, input_name6: y_scale,
        input_name7: y_zero_point
        })

print("The predicted output for the operation: qlinearconv")
print(y_pred)

y_pred = np.asarray(y_pred) #converting list into an array
print(y_pred.shape)

y_pred = np.squeeze(y_pred, axis=0)
print(y_pred.shape)

compare(y_actual, y_pred)