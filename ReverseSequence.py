#ReverseSequence (reversesequence_batch)
import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare

# Create one Input (ValueInfoProto)
x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [4, 4])
sequence_lens = helper.make_tensor_value_info('sequence_lens', TensorProto.INT64, [4,])

# Create one output (ValueInfoProto)
y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [4, 4])

# Create a node (NodeProto)
node_def = helper.make_node(
    'ReverseSequence',
    inputs=['x', 'sequence_lens'],
    outputs=['y'],
    time_axis=1,
    batch_axis=0,
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [x, sequence_lens],
    [y],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-ReverseSequence')

print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, 'onnx-ReverseSequence.onnx')
onnx.save(model_def, new_model_path)

print('The model is saved.')



# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, 'onnx-ReverseSequence.onnx')
onnx_model1 = onnx.load(model_path1)

print('The model is:\n{}'.format(onnx_model1))

import onnxruntime as rt
sess = rt.InferenceSession("onnx-ReverseSequence.onnx")
input_name = sess.get_inputs()[0].name
input_name1 = sess.get_inputs()[1].name
label_name = sess.get_outputs()[0].name

x = np.array([[0.0, 1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0, 7.0],
              [8.0, 9.0, 10.0, 11.0],
              [12.0, 13.0, 14.0, 15.0]], dtype=np.float32)
sequence_lens = np.array([1, 2, 3, 4], dtype=np.int64)

y_actual = np.array([[0.0, 1.0, 2.0, 3.0],
              [5.0, 4.0, 6.0, 7.0],
              [10.0, 9.0, 8.0, 11.0],
              [15.0, 14.0, 13.0, 12.0]], dtype=np.float32)

y_pred = sess.run(
        [], {input_name: x, input_name1: sequence_lens})

print("The predicted output for the operation: ReverseSequence (reversesequence_batch)", y_pred)

y_pred = np.asarray(y_pred) #converting list into an array
print(y_pred.shape)

y_pred = np.squeeze(y_pred, axis=0)
print(y_pred.shape)

compare(y_pred, y_actual)