#Pad(constant_pad)

import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare

# Create the inputs (ValueInfoProto)
x =  helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 3, 4, 5])
pads =  helper.make_tensor_value_info('pads', TensorProto.INT64, [8])
value =  helper.make_tensor_value_info('value', TensorProto.FLOAT, [1,])

# Create one output (ValueInfoProto)
y =  helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 3, 7, 12])


# Create a node (NodeProto)
node_def = helper.make_node(
    'Pad',
    inputs=['x', 'pads', 'value'],
    outputs=['y'],
    mode='constant'
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [x, pads, value],
    [y],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-Pad')
print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, 'Pad.onnx')
onnx.save(model_def, new_model_path)
print('The model is saved.')


# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, 'Pad.onnx')
onnx_model1 = onnx.load(model_path1)
print('The model is:\n{}'.format(onnx_model1))

def pad_impl(data, raw_pads, mode, constant_values=0.0):  # type: ignore

    input_rank = data.ndim
    if input_rank * 2 != raw_pads.size:
        raise Exception('The number of elements in raw_pads should be 2 * data_rank')

    # re-order to np.pad accepted order ((x1_begin, x1_end), (x2_begin, x2_end), ...)
    pad_width = ()
    for i in range(int(raw_pads.size / 2)):
        pad_width += ((raw_pads[i], raw_pads[i + input_rank])),  # type: ignore

    if mode == 'constant':
        y = np.pad(
            data,
            pad_width=pad_width,
            mode=mode,
            constant_values=constant_values,
        )
        return y

    y = np.pad(
        data,
        pad_width=pad_width,
        mode=mode,
    )

    return y


x = np.random.randn(1, 3, 4, 5).astype(np.float32)
pads = np.array([0, 0, 1, 3, 0, 0, 2, 4]).astype(np.int64)  # pad order [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
value = np.array([1.2]).astype(np.float32)
y_actual = pad_impl(
    x,
    pads,
    'constant',
    1.2
)


#Running the model using ONNX Runtime
import onnxruntime as rt
import numpy
sess = rt.InferenceSession("Pad.onnx")
input_name = sess.get_inputs()[0].name
input_name1 = sess.get_inputs()[1].name
input_name2 = sess.get_inputs()[2].name
label_name = sess.get_outputs()[0].name

y_pred = sess.run(
        [label_name], {input_name: x,
        input_name1: pads,
        input_name2: value
        })

print("The predicted output for the operation: Pad")
print(y_pred)

y_pred = np.asarray(y_pred) #converting list into an array
print(y_pred.shape)

y_pred = np.squeeze(y_pred, axis=0)
print(y_pred.shape)

compare(y_actual, y_pred)