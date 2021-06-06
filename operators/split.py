#split(1d)

import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare

# Create the inputs (ValueInfoProto)
x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [6])


# Create one output (ValueInfoProto)
y1 = helper.make_tensor_value_info('y1', TensorProto.FLOAT, [2])
y2 = helper.make_tensor_value_info('y2', TensorProto.FLOAT, [2])
y3 = helper.make_tensor_value_info('y3', TensorProto.FLOAT, [2])

# Create a node (NodeProto)
node_def = helper.make_node(
    'Split',
    inputs=['x'],
    outputs=['y1', 'y2', 'y3'],
    axis=0
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [x],
    [y1, y2, y3],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-split')
print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, '../onnx_generated_models/split.onnx')
onnx.save(model_def, new_model_path)
print('The model is saved.')


# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, '../onnx_generated_models/split.onnx')
onnx_model1 = onnx.load(model_path1)
print('The model is:\n{}'.format(onnx_model1))

x = np.array([1., 2., 3., 4., 5., 6.]).astype(np.float32)
y_actual = [np.array([1., 2.]).astype(np.float32), np.array([3., 4.]).astype(np.float32), np.array([5., 6.]).astype(np.float32)]
y_actual = np.array(y_actual)
#Running the model using ONNX Runtime
import onnxruntime as rt
import numpy
sess = rt.InferenceSession("../onnx_generated_models/split.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
label_name1 = sess.get_outputs()[1].name
label_name2 = sess.get_outputs()[2].name

y_pred = sess.run(
        [label_name, label_name1, label_name2], {input_name: x.astype(numpy.float32),
        })

print("The predicted output for the operation: split")
print(y_pred)

y_pred = np.asarray(y_pred) #converting list into an array
print(y_pred.shape)

compare(y_actual, y_pred)