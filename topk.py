#top_k
import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare

# Create the inputs (ValueInfoProto)
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [3, 4])
k = helper.make_tensor_value_info('k', TensorProto.INT64, [1,])

# Create one output (ValueInfoProto)
values = helper.make_tensor_value_info('values', TensorProto.FLOAT, [3, 3])
indices = helper.make_tensor_value_info('indices', TensorProto.INT64, [3, 3])

axis = 1
# Create a node (NodeProto)
node_def = helper.make_node(
    'TopK',
    inputs=['X', 'k'],
    outputs=['values', 'indices'],
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [X,k],
    [values, indices],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-topk')
print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

#model_def = onnx.utils.polish_model(model_def)
# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, 'topk.onnx')
onnx.save(model_def, new_model_path)
print('The model is saved.')


# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, 'topk.onnx')
onnx_model1 = onnx.load(model_path1)
print('The model is:\n{}'.format(onnx_model1))



def topk_sorted_implementation(X, k, axis, largest):  # type: ignore
    sorted_indices = np.argsort(X, axis=axis)
    sorted_values = np.sort(X, axis=axis)
    if largest:
        sorted_indices = np.flip(sorted_indices, axis=axis)
        sorted_values = np.flip(sorted_values, axis=axis)
    topk_sorted_indices = np.take(sorted_indices, np.arange(k), axis=axis)
    topk_sorted_values = np.take(sorted_values, np.arange(k), axis=axis)
    return topk_sorted_values, topk_sorted_indices

largest = 1
k = 3
X = np.array([
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [8, 9, 10, 11],
], dtype=np.float32)
K = np.array([k], dtype=np.int64)
values_ref, indices_ref = topk_sorted_implementation(X, k, axis, largest)


#Running the model using ONNX Runtime
import onnxruntime as rt
import numpy
sess = rt.InferenceSession("topk.onnx")
input_name = sess.get_inputs()[0].name
input_name1 = sess.get_inputs()[1].name
label_name = sess.get_outputs()[0].name
label_name1 = sess.get_outputs()[1].name


y_pred = sess.run(
        [], {input_name: X.astype(numpy.float32),
        input_name1: K.astype(numpy.int64)})

print("The predicted output for the operation: top_k")
print(y_pred)

y_pred = np.asarray(y_pred) #converting list into an array
print(y_pred.shape)

y_pred = y_pred[0, :, :]
print(y_pred.shape)

compare(values_ref, y_pred)