#quantizelinear

import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare

# Create the inputs (ValueInfoProto)
x =  helper.make_tensor_value_info('x', TensorProto.FLOAT, [6,])
y_scale =  helper.make_tensor_value_info('y_scale', TensorProto.FLOAT, [1,])
y_zero_point =  helper.make_tensor_value_info('y_zero_point', TensorProto.UINT8, [1,])


# Create one output (ValueInfoProto)
y =  helper.make_tensor_value_info('y', TensorProto.UINT8, [6,])


# Create a node (NodeProto)
node_def = helper.make_node(
    'QuantizeLinear',
    inputs=['x', 'y_scale', 'y_zero_point'],
    outputs=['y'],
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [x, y_scale, y_zero_point],
    [y],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-quantizelinear')
print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, '../onnx_generated_models/quantizelinear.onnx')
onnx.save(model_def, new_model_path)
print('The model is saved.')


# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, '../onnx_generated_models/quantizelinear.onnx')
onnx_model1 = onnx.load(model_path1)
print('The model is:\n{}'.format(onnx_model1))

x = np.array([0, 2, 3, 1000, -254, -1000]).astype(np.float32)
y_scale = np.array([2], dtype=np.float32)
y_zero_point = np.array([128], dtype=np.uint8)

y_actual = np.array([128, 129, 130, 255, 1, 0]).astype(np.uint8)


#Running the model using ONNX Runtime
import onnxruntime as rt
import numpy
sess = rt.InferenceSession("../onnx_generated_models/quantizelinear.onnx")
input_name = sess.get_inputs()[0].name
input_name1 = sess.get_inputs()[1].name
input_name2 = sess.get_inputs()[2].name
label_name = sess.get_outputs()[0].name

y_pred = sess.run(
        [label_name], {input_name: x,
        input_name1: y_scale, input_name2: y_zero_point,
        })

print("The predicted output for the operation: quantizelinear")
print(y_pred)

y_pred = np.asarray(y_pred) #converting list into an array
print(y_pred.shape)

y_pred = np.squeeze(y_pred, axis=0)
print(y_pred.shape)

compare(y_actual, y_pred)