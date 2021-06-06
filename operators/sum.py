#sum

import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare

# Create the inputs (ValueInfoProto)
x1 = helper.make_tensor_value_info('x1', TensorProto.FLOAT, [3,])
x2 = helper.make_tensor_value_info('x2', TensorProto.FLOAT, [3,])
x3 = helper.make_tensor_value_info('x3', TensorProto.FLOAT, [3,])


# Create one output (ValueInfoProto)
y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [3,])

# Create a node (NodeProto)
node_def = helper.make_node(
    'Sum',
    inputs=['x1', 'x2', 'x3'],
    outputs=['y'],
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [x1, x2, x3],
    [y],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-sum')
print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, '../onnx_generated_models/sum.onnx')
onnx.save(model_def, new_model_path)
print('The model is saved.')


# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, '../onnx_generated_models/sum.onnx')
onnx_model1 = onnx.load(model_path1)
print('The model is:\n{}'.format(onnx_model1))

x1 = np.array([3, 0, 2]).astype(np.float32)
x2 = np.array([1, 3, 4]).astype(np.float32)
x3 = np.array([2, 6, 6]).astype(np.float32)
y_actual = np.array([6, 9, 12]).astype(np.float32)

#Running the model using ONNX Runtime
import onnxruntime as rt
import numpy
sess = rt.InferenceSession("../onnx_generated_models/sum.onnx")
input_name = sess.get_inputs()[0].name
input_name1 = sess.get_inputs()[1].name
input_name2 = sess.get_inputs()[2].name
label_name = sess.get_outputs()[0].name


y_pred = sess.run(
        [label_name], {input_name: x1.astype(numpy.float32),
        input_name1: x2.astype(numpy.float32), input_name2: x3.astype(numpy.float32)
        })

print("The predicted output for the operation: sum")
print(y_pred)

y_pred = np.asarray(y_pred) #converting list into an array
print(y_pred.shape)

y_pred = np.squeeze(y_pred, axis=0)
print(y_pred.shape)

compare(y_actual, y_pred)