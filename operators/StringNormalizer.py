#StringNormalizer(monday_casesensintive_lower)

import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare

# Create the inputs (ValueInfoProto)
x = helper.make_tensor_value_info('x', TensorProto.STRING, [4])


# Create one output (ValueInfoProto)
y = helper.make_tensor_value_info('y', TensorProto.STRING, [3])

stopwords = [u'monday']
# Create a node (NodeProto)
node_def = helper.make_node(
    'StringNormalizer',
    inputs=['x'],
    outputs=['y'],
    case_change_action='LOWER',
    is_case_sensitive=1,
    stopwords=stopwords
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [x],
    [y],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-StringNormalizer')
print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, '../onnx_generated_models/StringNormalizer.onnx')
onnx.save(model_def, new_model_path)
print('The model is saved.')


# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, '../onnx_generated_models/StringNormalizer.onnx')
onnx_model1 = onnx.load(model_path1)
print('The model is:\n{}'.format(onnx_model1))

x = np.array([u'monday', u'tuesday', u'wednesday', u'thursday']).astype(np.object)
y_actual = np.array([u'tuesday', u'wednesday', u'thursday']).astype(np.object)

#Running the model using ONNX Runtime
import onnxruntime as rt
import numpy
sess = rt.InferenceSession("../onnx_generated_models/StringNormalizer.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name


y_pred = sess.run(
        [label_name], {input_name: x.astype(numpy.object),
        })

print("The predicted output for the operation: StringNormalizer")
print(y_pred)

y_pred = np.asarray(y_pred) #converting list into an array
print(y_pred.shape)

y_pred = np.squeeze(y_pred, axis=0)
print(y_pred.shape)

compare(y_actual, y_pred)