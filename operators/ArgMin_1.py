import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare

# Create one Input (ValueInfoProto)
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [2, 2])

# Create one output (ValueInfoProto)
Y = helper.make_tensor_value_info('Y', TensorProto.INT64, [1, 2])

keepdims = 1
# Create a node (NodeProto)
node_def = helper.make_node(
    'ArgMin',
    inputs=['X'],
    outputs=['Y'],
    keepdims=keepdims
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [X],
    [Y],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-ArgMin_1')

print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, '../onnx_generated_models/onnx-ArgMin_1.onnx')
onnx.save(model_def, new_model_path)

print('The model is saved.')



# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, '../onnx_generated_models/onnx-ArgMin_1.onnx')
onnx_model1 = onnx.load(model_path1)

print('The model is:\n{}'.format(onnx_model1))

import onnxruntime as rt
sess = rt.InferenceSession("../onnx_generated_models/onnx-ArgMin_1.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

x = np.array([[2, 1], [3, 10]], dtype=np.float32)

def argmin_use_numpy(data, axis=0, keepdims=1):  # type: (np.ndarray, int, int) -> (np.ndarray)
    result = np.argmin(data, axis=axis)
    if (keepdims == 1):
        result = np.expand_dims(result, axis)
    return result.astype(np.float32)

y_actual = argmin_use_numpy(x, keepdims=keepdims)


y_pred = sess.run(
        [], {input_name: x})

print("The predicted output for the operation: ArgMin_1")
print(y_pred)


y_pred = np.asarray(y_pred) #converting list into an array
print(y_pred.shape)

y_pred = np.squeeze(y_pred, axis=0)
print(y_pred.shape)

compare(y_pred, y_actual)
#print(pred)
#print(pred.shape)
