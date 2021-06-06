#EyeLike (populate_off_main_diagonal)
import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare

# Create one Input (ValueInfoProto)
a = helper.make_tensor_value_info('a', TensorProto.FLOAT, [2, 3, 4, 5])

# Create one output (ValueInfoProto)
b = helper.make_tensor_value_info('b', TensorProto.FLOAT, [24, 5])

shape = (2, 3, 4, 5)
# Create a node (NodeProto)
for i in range(len(shape)):
    node_def = onnx.helper.make_node(
        'Flatten',
        inputs=['a'],
        outputs=['b'],
        axis=i,
    )

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [a],
    [b],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-flatten')

print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, '../onnx_generated_models/onnx-flatten.onnx')
onnx.save(model_def, new_model_path)

print('The model is saved.')



# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, '../onnx_generated_models/onnx-flatten.onnx')
onnx_model1 = onnx.load(model_path1)

print('The model is:\n{}'.format(onnx_model1))

import onnxruntime as rt
sess = rt.InferenceSession("../onnx_generated_models/onnx-flatten.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name


a = np.random.random_sample(shape).astype(np.float32)

new_shape = (1, -1) if i == 0 else (np.prod(shape[0:i]).astype(int), -1)
b = np.reshape(a, new_shape)

y_pred = sess.run(
        [], {input_name: a})

print("The predicted output for the operation: flatten", y_pred)

y_pred = np.asarray(y_pred) #converting list into an array
print(y_pred.shape)

y_pred = np.squeeze(y_pred, axis=0)
print(y_pred.shape)

compare(y_pred, b)

