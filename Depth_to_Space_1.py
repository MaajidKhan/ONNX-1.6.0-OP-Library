#DepthToSpace (crd_mode_example)
import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare

# Create the inputs (ValueInfoProto)
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 8, 2, 3])

# Create one output (ValueInfoProto)
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2, 4, 6])

# Create a node (NodeProto)
node_def = helper.make_node(
    'DepthToSpace',
    inputs=['X'],
    outputs=['Y'],
    blocksize=2,
    mode='CRD'
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [X],
    [Y],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-Depth_to_space_1')
print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

#model_def = onnx.utils.polish_model(model_def)
# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, 'Depth_to_space_1.onnx')
onnx.save(model_def, new_model_path)
print('The model is saved.')


# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, 'Depth_to_space_1.onnx')
onnx_model1 = onnx.load(model_path1)
print('The model is:\n{}'.format(onnx_model1))


# (1, 8, 2, 3) input tensor
x = np.array([[[[0., 1., 2.],
                [3., 4., 5.]],
               [[9., 10., 11.],
                [12., 13., 14.]],
               [[18., 19., 20.],
                [21., 22., 23.]],
               [[27., 28., 29.],
                [30., 31., 32.]],
               [[36., 37., 38.],
                [39., 40., 41.]],
               [[45., 46., 47.],
                [48., 49., 50.]],
               [[54., 55., 56.],
                [57., 58., 59.]],
               [[63., 64., 65.],
                [66., 67., 68.]]]]).astype(np.float32)

# (1, 2, 4, 6) output tensor
y_actual = np.array([[[[0., 9., 1., 10., 2., 11.],
                [18., 27., 19., 28., 20., 29.],
                [3., 12., 4., 13., 5., 14.],
                [21., 30., 22., 31., 23., 32.]],
               [[36., 45., 37., 46., 38., 47.],
                [54., 63., 55., 64., 56., 65.],
                [39., 48., 40., 49., 41., 50.],
                [57., 66., 58., 67., 59., 68.]]]]).astype(np.float32)

#Running the model using ONNX Runtime
import onnxruntime as rt
import numpy
sess = rt.InferenceSession("Depth_to_space_1.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name


y_pred = sess.run(
        [label_name], {input_name: x.astype(numpy.float32)})

print("The predicted output for the operation: DepthToSpace")
print(y_pred)

y_pred = np.asarray(y_pred) #converting list into an array
print(y_pred.shape)

y_pred = np.squeeze(y_pred, axis=0)
print(y_pred.shape)

compare(y_actual, y_pred)