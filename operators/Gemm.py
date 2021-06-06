#Gemm [all_attributes]
import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare

# Create the inputs (ValueInfoProto)
a = helper.make_tensor_value_info('a', TensorProto.FLOAT, [4, 3])
b = helper.make_tensor_value_info('b', TensorProto.FLOAT, [5, 4])
c = helper.make_tensor_value_info('c', TensorProto.FLOAT, [1, 5])


# Create one output (ValueInfoProto)
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [3, 5])

# Create a node (NodeProto)
node_def = helper.make_node(
    'Gemm',
    inputs=['a', 'b', 'c'],
    outputs=['Y'],
    alpha=0.25,
    beta=0.35,
    transA=1,
    transB=1
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [a, b, c],
    [Y],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-gemm')
print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

#model_def = onnx.utils.polish_model(model_def)
# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, '../onnx_generated_models/gemm.onnx')
onnx.save(model_def, new_model_path)
print('The model is saved.')


# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, '../onnx_generated_models/gemm.onnx')
onnx_model1 = onnx.load(model_path1)
print('The model is:\n{}'.format(onnx_model1))


a = np.random.ranf([4, 3]).astype(np.float32)
b = np.random.ranf([5, 4]).astype(np.float32)
c = np.random.ranf([1, 5]).astype(np.float32)

def gemm_reference_implementation(A, B, C=None, alpha=1., beta=1., transA=0,
                                  transB=0):  # type: (np.ndarray, np.ndarray, Optional[np.ndarray], float, float, int, int) -> np.ndarray
    A = A if transA == 0 else A.T
    B = B if transB == 0 else B.T
    C = C if C is not None else np.array(0)

    Y = alpha * np.dot(A, B) + beta * C

    return Y

y_actual = gemm_reference_implementation(a, b, c, transA=1, transB=1, alpha=0.25, beta=0.35)

#Running the model using ONNX Runtime
import onnxruntime as rt
import numpy
sess = rt.InferenceSession("../onnx_generated_models/gemm.onnx")
input_name = sess.get_inputs()[0].name
input_name1 = sess.get_inputs()[1].name
input_name2 = sess.get_inputs()[2].name
label_name = sess.get_outputs()[0].name


y_pred = sess.run(
        [label_name], {input_name: a.astype(numpy.float32),
        input_name1: b.astype(numpy.float32), input_name2: c.astype(numpy.float32)})

print("The predicted output for the operation: Gemm")
print(y_pred)

y_pred = np.asarray(y_pred) #converting list into an array
print(y_pred.shape)

y_pred = np.squeeze(y_pred, axis=0)
print(y_pred.shape)

y_pred = np.round(y_pred, 6)
y_actual = np.round(y_actual, 6)
compare(y_actual, y_pred)