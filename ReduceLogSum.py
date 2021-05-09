#ReduceLogSum (keepdims)
import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare

# Create one Input (ValueInfoProto)
data = helper.make_tensor_value_info('data', TensorProto.FLOAT, [3, 4, 5])

# Create one output (ValueInfoProto)
reduced = helper.make_tensor_value_info('reduced', TensorProto.FLOAT, [1, 1, 1])


# Create a node (NodeProto)
node_def = helper.make_node(
    'ReduceLogSum',
    inputs=['data'],
    outputs=["reduced"]
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [data],
    [reduced],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-ReduceLogSum')

print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, 'onnx-ReduceLogSum.onnx')
onnx.save(model_def, new_model_path)

print('The model is saved.')



# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, 'onnx-ReduceLogSum.onnx')
onnx_model1 = onnx.load(model_path1)

print('The model is:\n{}'.format(onnx_model1))

import onnxruntime as rt
sess = rt.InferenceSession("onnx-ReduceLogSum.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

data = np.random.ranf([3, 4, 5]).astype(np.float32)
y_actual = np.log(np.sum(data, keepdims=True))

y_pred = sess.run(
        [], {input_name: data})

print("The predicted output for the operation: ReduceLogSum (keepdims)", y_pred)

y_pred = np.asarray(y_pred) #converting list into an array
print(y_pred.shape)

y_pred = np.squeeze(y_pred, axis=0)
print(y_pred.shape)

y_pred1 = np.round(y_pred, 6)
y_actual1 = np.round(y_actual, 6)
compare(y_pred1, y_actual1)