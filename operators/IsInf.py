#IsInf (infinity)
import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare

# Create one Input (ValueInfoProto)
x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [6])

# Create one output (ValueInfoProto)
y = helper.make_tensor_value_info('y', TensorProto.BOOL, [6])

# Create a node (NodeProto)
node_def = helper.make_node(
    'IsInf',
     inputs=['x'],
     outputs=['y'],
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [x],
    [y],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-IsInf')

print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, '../onnx_generated_models/onnx-IsInf.onnx')
onnx.save(model_def, new_model_path)

print('The model is saved.')



# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, '../onnx_generated_models/onnx-IsInf.onnx')
onnx_model1 = onnx.load(model_path1)

print('The model is:\n{}'.format(onnx_model1))

import onnxruntime as rt
sess = rt.InferenceSession("../onnx_generated_models/onnx-IsInf.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

x = np.array([-1.2, np.nan, np.inf, 2.8, np.NINF, np.inf],
             dtype=np.float32)
y_actual = np.isinf(x)

y_pred = sess.run(
        [], {input_name: x
       })

print("The predicted output for the operation: IsInf", y_pred)

y_pred = np.asarray(y_pred) #converting list into an array
print(y_pred.shape)

y_pred = np.squeeze(y_pred, axis=0)
print(y_pred.shape)

compare(y_pred, y_actual)
