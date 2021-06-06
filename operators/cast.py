#cast

import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE

# Create the inputs (ValueInfoProto)
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [3, 4])

# Create one output (ValueInfoProto)
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [3, 4])

shape = (3, 4)
test_cases = [
    ('FLOAT', 'FLOAT16'),
    ('FLOAT', 'DOUBLE'),
    ('FLOAT16', 'FLOAT'),
    ('FLOAT16', 'DOUBLE'),
    ('DOUBLE', 'FLOAT'),
    ('DOUBLE', 'FLOAT16'),
    ('FLOAT', 'STRING'),
    ('STRING', 'FLOAT'),
]

for from_type, to_type in test_cases:
    if 'STRING' != from_type:
        input = np.random.random_sample(shape).astype(
            TENSOR_TYPE_TO_NP_TYPE[getattr(TensorProto, from_type)])
        if ('STRING' == to_type):
            # Converting input to str, then give it np.object dtype for generating script
            ss = []
            for i in input.flatten():
                s = str(i).encode('utf-8')
                su = s.decode('utf-8')
                ss.append(su)

            y_actual = np.array(ss).astype(np.object).reshape([3, 4])
        else:
            y_actual = input.astype(TENSOR_TYPE_TO_NP_TYPE[getattr(TensorProto, to_type)])
    else:
        input = np.array([u'0.47892547', u'0.48033667', u'0.49968487', u'0.81910545',
            u'0.47031248', u'0.816468', u'0.21087195', u'0.7229038',
            u'0.34589098', u'0.76567456', u'0.65786723', u'0.56789876'], dtype=np.dtype(np.object)).reshape([3, 4])
        y_actual = input.astype(TENSOR_TYPE_TO_NP_TYPE[getattr(TensorProto, to_type)])

# Create a node (NodeProto)
node_def = helper.make_node(
    'Cast',
    inputs=['X'],
    outputs=['Y'],
    to=getattr(TensorProto, to_type),
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [X],
    [Y],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-cast')
print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, '../onnx_generated_models/cast.onnx')
onnx.save(model_def, new_model_path)
print('The model is saved.')


# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, '../onnx_generated_models/cast.onnx')
onnx_model1 = onnx.load(model_path1)
print('The model is:\n{}'.format(onnx_model1))


#Running the model using ONNX Runtime
import onnxruntime as rt
import numpy
sess = rt.InferenceSession("../onnx_generated_models/cast.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

y_pred = sess.run(
        [label_name], {input_name: input.astype(numpy.float32)})

print("The predicted output for the operation: Cast")
print(y_pred)

y_pred = np.asarray(y_pred) #converting list into an array
print(y_pred.shape)

y_pred = np.squeeze(y_pred, axis=0)
print(y_pred.shape)

compare(y_actual, y_pred)