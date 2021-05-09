#DynamicQuantizeLinear
import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare

# Create one Input (ValueInfoProto)
x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [6])

# Create one output (ValueInfoProto)
y = helper.make_tensor_value_info('y', TensorProto.UINT8, [6])
y_scale = helper.make_tensor_value_info('y_scale', TensorProto.FLOAT, [])
y_zero_point = helper.make_tensor_value_info('y_zero_point', TensorProto.UINT8, [])

# Create a node (NodeProto)
node_def = helper.make_node(
    'DynamicQuantizeLinear',
    inputs=['x'],
    outputs=['y', 'y_scale', 'y_zero_point'],
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [x],
    [y, y_scale, y_zero_point],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-DynamicQuantizeLinear')

print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, 'onnx-DynamicQuantizeLinear.onnx')
onnx.save(model_def, new_model_path)

print('The model is saved.')



# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, 'onnx-DynamicQuantizeLinear.onnx')
onnx_model1 = onnx.load(model_path1)

print('The model is:\n{}'.format(onnx_model1))

import onnxruntime as rt
sess = rt.InferenceSession("onnx-DynamicQuantizeLinear.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
label_name1 = sess.get_outputs()[1].name
label_name2 = sess.get_outputs()[2].name

# expected scale 0.0196078438 and zero point 153
X = np.array([0, 2, -3, -2.5, 1.34, 0.5]).astype(np.float32)
x_min = np.minimum(0, np.min(X))
x_max = np.maximum(0, np.max(X))
Y_Scale = np.float32((x_max - x_min) / (255 - 0))  # uint8 -> [0, 255]
Y_ZeroPoint = np.clip(round((0 - x_min) / Y_Scale), 0, 255).astype(np.uint8)
y_actual = np.clip(np.round(X / Y_Scale) + Y_ZeroPoint, 0, 255).astype(np.uint8)

y_pred = sess.run(
        [], {input_name: X})

print("The predicted output for the operation: DynamicQuantizeLinear", y_pred)

y_pred = np.asarray(y_pred) #converting list into an array
print(y_pred.shape)



compare(y_pred[0], y_actual)
compare(y_pred[1], Y_Scale)
compare(y_pred[2], Y_ZeroPoint)