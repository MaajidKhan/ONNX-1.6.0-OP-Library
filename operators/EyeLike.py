#EyeLike (populate_off_main_diagonal)
import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare

# Create one Input (ValueInfoProto)
x = helper.make_tensor_value_info('x', TensorProto.INT32, [4, 5])

# Create one output (ValueInfoProto)
y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [4, 5])

shape = (4, 5)
off_diagonal_offset = 1
# Create a node (NodeProto)
node_def = helper.make_node(
    'EyeLike',
    inputs=['x'],
    outputs=['y'],
    k=off_diagonal_offset,
    dtype=onnx.TensorProto.FLOAT,
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [x],
    [y],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-EyeLike')

print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, '../onnx_generated_models/onnx-EyeLike.onnx')
onnx.save(model_def, new_model_path)

print('The model is saved.')



# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, '../onnx_generated_models/onnx-EyeLike.onnx')
onnx_model1 = onnx.load(model_path1)

print('The model is:\n{}'.format(onnx_model1))

import onnxruntime as rt
sess = rt.InferenceSession("../onnx_generated_models/onnx-EyeLike.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

x = np.random.randint(0, 100, size=shape, dtype=np.int32)
y_actual = np.eye(shape[0], shape[1], k=off_diagonal_offset, dtype=np.float32)

y_pred = sess.run(
        [], {input_name: x})

print("The predicted output for the operation: EyeLike (populate_off_main_diagonal)", y_pred)

y_pred = np.asarray(y_pred) #converting list into an array
print(y_pred.shape)

y_pred = np.squeeze(y_pred, axis=0)
print(y_pred.shape)

compare(y_pred, y_actual)

