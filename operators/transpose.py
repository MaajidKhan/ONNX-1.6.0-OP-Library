#Transpose [default]
import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare

# Create the inputs (ValueInfoProto)
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [2, 3, 4])

# Create one output (ValueInfoProto)
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [4, 3, 2])

# Create a node (NodeProto)
node_def = helper.make_node(
    'Transpose',
    inputs=['X'],
    outputs=['Y'],
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [X],
    [Y],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-transpose')
print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

#model_def = onnx.utils.polish_model(model_def)
# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, '../onnx_generated_models/transpose.onnx')
onnx.save(model_def, new_model_path)
print('The model is saved.')


# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, '../onnx_generated_models/transpose.onnx')
onnx_model1 = onnx.load(model_path1)
print('The model is:\n{}'.format(onnx_model1))


shape = (2, 3, 4)
x = np.random.random_sample(shape).astype(np.float32)
y_actual = np.transpose(x)

#Running the model using ONNX Runtime
import onnxruntime as rt
import numpy
sess = rt.InferenceSession("../onnx_generated_models/transpose.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name


y_pred = sess.run(
        [label_name], {input_name: x.astype(numpy.float32)})

print("The predicted output for the operation: Transpose")
print(y_pred)

y_pred = np.asarray(y_pred) #converting list into an array
print(y_pred.shape)

y_pred = np.squeeze(y_pred, axis=0)
print(y_pred.shape)

compare(y_actual, y_pred)