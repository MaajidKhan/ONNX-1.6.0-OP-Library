#convtranspose
import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare

# Create the inputs (ValueInfoProto)
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 1, 3, 3])
W = helper.make_tensor_value_info('W', TensorProto.FLOAT, [1, 2, 3, 3])

# Create one output (ValueInfoProto)
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2, 5, 5])


# Create a node (NodeProto)
node_def = helper.make_node(
    "ConvTranspose",
    ["X", "W"],
    ["Y"]
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [X, W],
    [Y],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-ConvTranspose')

print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, '../onnx_generated_models/onnx-ConvTranspose.onnx')
onnx.save(model_def, new_model_path)

print('The model is saved.')



# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, '../onnx_generated_models/onnx-ConvTranspose.onnx')
onnx_model1 = onnx.load(model_path1)

print('The model is:\n{}'.format(onnx_model1))



import onnxruntime as rt
import numpy as np
sess = rt.InferenceSession("../onnx_generated_models/onnx-ConvTranspose.onnx")
input_name = sess.get_inputs()[0].name
input_name1 = sess.get_inputs()[1].name
label_name = sess.get_outputs()[0].name

x = np.array([[[[0., 1., 2.],  # (1, 1, 3, 3)
                [3., 4., 5.],
                [6., 7., 8.]]]]).astype(np.float32)

W = np.array([[[[1., 1., 1.],  # (1, 2, 3, 3)
                [1., 1., 1.],
                [1., 1., 1.]],
               [[1., 1., 1.],
                [1., 1., 1.],
                [1., 1., 1.]]]]).astype(np.float32)

y_actual = np.array([[[[0., 1., 3., 3., 2.],  # (1, 2, 5, 5)
                [3., 8., 15., 12., 7.],
                [9., 21., 36., 27., 15.],
                [9., 20., 33., 24., 13.],
                [6., 13., 21., 15., 8.]],

               [[0., 1., 3., 3., 2.],
                [3., 8., 15., 12., 7.],
                [9., 21., 36., 27., 15.],
                [9., 20., 33., 24., 13.],
                [6., 13., 21., 15., 8.]]]]).astype(np.float32)


y_pred = sess.run(
        [], {input_name: x,
        input_name1: W })


print("The predicted output for the operation: ConvTranspose")
print(y_pred)

y_pred = np.asarray(y_pred) #converting list into an array
print(y_pred.shape)

y_pred = np.squeeze(y_pred, axis=0)
print(y_pred.shape)

compare(y_actual, y_pred)