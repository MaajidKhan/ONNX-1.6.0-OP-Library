#DequantizeLinear

import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare

# Create the inputs (ValueInfoProto)
x =  helper.make_tensor_value_info('x', TensorProto.UINT8, [4,])
x_scale =  helper.make_tensor_value_info('x_scale', TensorProto.FLOAT, [1,])
x_zero_point =  helper.make_tensor_value_info('x_zero_point', TensorProto.UINT8, [1,])


# Create one output (ValueInfoProto)
y =  helper.make_tensor_value_info('y', TensorProto.FLOAT, [4,])



# Create a node (NodeProto)
node_def = helper.make_node(
    'DequantizeLinear',
    inputs=['x', 'x_scale', 'x_zero_point'],
    outputs=['y'],
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [x, x_scale, x_zero_point],
    [y],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-DequantizeLinear')
print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, 'DequantizeLinear.onnx')
onnx.save(model_def, new_model_path)
print('The model is saved.')


# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, 'DequantizeLinear.onnx')
onnx_model1 = onnx.load(model_path1)
print('The model is:\n{}'.format(onnx_model1))

x = np.array([0, 3, 128, 255]).astype(np.uint8)
x_scale = np.array([2]).astype(np.float32)  
x_zero_point = np.array([128]).astype(np.uint8)
y_actual = np.array([-256, -250, 0, 254], dtype=np.float32)

#Running the model using ONNX Runtime
import onnxruntime as rt
import numpy
sess = rt.InferenceSession("DequantizeLinear.onnx")
input_name = sess.get_inputs()[0].name
input_name1 = sess.get_inputs()[1].name
input_name2 = sess.get_inputs()[2].name
label_name = sess.get_outputs()[0].name

y_pred = sess.run(
        [label_name], {input_name: x,
        input_name1: x_scale, input_name2: x_zero_point
        })

print("The predicted output for the operation: DequantizeLinear")
print(y_pred)

y_pred = np.asarray(y_pred) #converting list into an array
print(y_pred.shape)

y_pred = np.squeeze(y_pred, axis=0)
print(y_pred.shape)

compare(y_actual, y_pred)