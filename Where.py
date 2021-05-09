#Where(long)

import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare

# Create the inputs (ValueInfoProto)
condition =  helper.make_tensor_value_info('condition', TensorProto.BOOL, [2, 2])
x1 =  helper.make_tensor_value_info('x1', TensorProto.INT64, [2, 2])
x2 =  helper.make_tensor_value_info('x2', TensorProto.INT64, [2, 2])



# Create one output (ValueInfoProto)
y = helper.make_tensor_value_info('y', TensorProto.INT64, [2, 2])


# Create a node (NodeProto)
node_def = helper.make_node(
    'Where',
    inputs=['condition', 'x1', 'x2'],
    outputs=['y'],
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [condition, x1, x2],
    [y],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-Where')
print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, 'Where.onnx')
onnx.save(model_def, new_model_path)
print('The model is saved.')


# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, 'Where.onnx')
onnx_model1 = onnx.load(model_path1)
print('The model is:\n{}'.format(onnx_model1))

condition = np.array([[1, 0], [1, 1]], dtype=np.bool)
x1 = np.array([[1, 2], [3, 4]], dtype=np.int64)
x2 = np.array([[9, 8], [7, 6]], dtype=np.int64)
y_actual = np.where(condition, x1, x2)  # expected output [[1, 8], [3, 4]]


#Running the model using ONNX Runtime
import onnxruntime as rt
import numpy
sess = rt.InferenceSession("Where.onnx")
input_name = sess.get_inputs()[0].name
input_name1 = sess.get_inputs()[1].name
input_name2 = sess.get_inputs()[2].name
label_name = sess.get_outputs()[0].name

y_pred = sess.run(
        [label_name], {input_name: condition.astype(numpy.bool),
        input_name1: x1.astype(numpy.int64),
        input_name2: x2.astype(numpy.int64)
        })

print("The predicted output for the operation: Where")
print(y_pred)

y_pred = np.asarray(y_pred) #converting list into an array
print(y_pred.shape)

y_pred = np.squeeze(y_pred, axis=0)
print(y_pred.shape)

compare(y_actual, y_pred)