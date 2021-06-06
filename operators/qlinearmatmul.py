#qlinearmatmul

import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare

# Create the inputs (ValueInfoProto)
a =  helper.make_tensor_value_info('a', TensorProto.UINT8, [2, 4])
a_scale =  helper.make_tensor_value_info('a_scale', TensorProto.FLOAT, [1,])
a_zero_point =  helper.make_tensor_value_info('a_zero_point', TensorProto.UINT8, [1,])
b =  helper.make_tensor_value_info('b', TensorProto.UINT8, [4, 3])
b_scale =  helper.make_tensor_value_info('b_scale', TensorProto.FLOAT, [1,])
b_zero_point =  helper.make_tensor_value_info('b_zero_point', TensorProto.UINT8, [1,])
y_scale =  helper.make_tensor_value_info('y_scale', TensorProto.FLOAT, [1,])
y_zero_point =  helper.make_tensor_value_info('y_zero_point', TensorProto.UINT8, [1,])

# Create one output (ValueInfoProto)
output =  helper.make_tensor_value_info('output', TensorProto.UINT8, [2, 3])


# Create a node (NodeProto)
node_def = helper.make_node(
    'QLinearMatMul',
    inputs=['a', 'a_scale', 'a_zero_point', 'b', 'b_scale', 'b_zero_point', 'y_scale', 'y_zero_point'],
    outputs=['output'],
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point],
    [output],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-qlinearmatmul')
print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, '../onnx_generated_models/qlinearmatmul.onnx')
onnx.save(model_def, new_model_path)
print('The model is saved.')


# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, '../onnx_generated_models/qlinearmatmul.onnx')
onnx_model1 = onnx.load(model_path1)
print('The model is:\n{}'.format(onnx_model1))


#2D
a = np.array([[208, 236, 0, 238],
    [3, 214, 255, 29], ], dtype=np.uint8)

a_scale = np.array([0.0066], dtype=np.float32)
a_zero_point = np.array([113], dtype=np.uint8)

b = np.array([[152, 51, 244],
    [60, 26, 255],
    [0, 127, 246],
    [127, 254, 247]], dtype=np.uint8)

b_scale = np.array([0.00705], dtype=np.float32)
b_zero_point = np.array([114], dtype=np.uint8)

y_scale = np.array([0.0107], dtype=np.float32)
y_zero_point = np.array([118], dtype=np.uint8)

y_actual = np.array([[168, 115, 255],
    [1, 66, 151], ], dtype=np.uint8)



#Running the model using ONNX Runtime
import onnxruntime as rt
import numpy
sess = rt.InferenceSession("../onnx_generated_models/qlinearmatmul.onnx")
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
        [label_name], {input_name: a,
        input_name1: a_scale, input_name2: a_zero_point,
        input_name3: b, input_name4: b_scale,
        input_name5: b_zero_point, input_name6: y_scale,
        input_name7: y_zero_point
        })

print("The predicted output for the operation: qlinearmatmul")
print(y_pred)

y_pred = np.asarray(y_pred) #converting list into an array
print(y_pred.shape)

y_pred = np.squeeze(y_pred, axis=0)
print(y_pred.shape)

compare(y_actual, y_pred)