import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare

# Create one Input (ValueInfoProto)
X1 = helper.make_tensor_value_info('X1', TensorProto.FLOAT, [3, 4, 5])
X2 = helper.make_tensor_value_info('X2', TensorProto.FLOAT, [3, 4, 5])

# Create one output (ValueInfoProto)
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [3, 4, 5])


# Create a node (NodeProto)
node_def = helper.make_node(
    'Add',
    inputs=['X1','X2'],
    outputs=['Y'],
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [X1,X2],
    [Y],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-Add')

print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, 'onnx-Add.onnx')
onnx.save(model_def, new_model_path)

print('The model is saved.')



# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, 'onnx-Add.onnx')
onnx_model1 = onnx.load(model_path1)

print('The model is:\n{}'.format(onnx_model1))

import onnxruntime as rt
sess = rt.InferenceSession("onnx-Add.onnx")
input_name = sess.get_inputs()[0].name
input_name1 = sess.get_inputs()[1].name
label_name = sess.get_outputs()[0].name

x1 = np.random.randn(3, 4, 5).astype(np.float32)
x2 = np.random.randn(3, 4, 5).astype(np.float32)
y_actual = x1 + x2
y_pred = sess.run(
        [], {input_name: x1,
        input_name1: x2})

print("The predicted output for the operation: Add")
print(y_pred)


y_pred = np.asarray(y_pred) #converting list into an array
print(y_pred.shape)

y_pred = np.squeeze(y_pred, axis=0)
print(y_pred.shape)


compare(y_pred, y_actual)
#print(pred)
#print(pred.shape)
