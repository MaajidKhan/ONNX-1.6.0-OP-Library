#averagepool_1d_default

import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
import itertools
from Compare_output import compare

# Create the inputs (ValueInfoProto)
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 3, 32])

# Create one output (ValueInfoProto)
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 3, 31])

# Create a node (NodeProto)
node_def = helper.make_node(
    'AveragePool',
    inputs=['X'],
    outputs=['Y'],
    kernel_shape=[2],
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [X],
    [Y],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-averagepool1')
print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, '../onnx_generated_models/averagepool1.onnx')
onnx.save(model_def, new_model_path)
print('The model is saved.')


# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, '../onnx_generated_models/averagepool1.onnx')
onnx_model1 = onnx.load(model_path1)
print('The model is:\n{}'.format(onnx_model1))

def get_output_shape(auto_pad,  # type: Text
                     input_spatial_shape,  # type: Sequence[int]
                     kernel_spatial_shape,  # type: Sequence[int]
                     strides_spatial  # type: Sequence[int]
                     ):  # type: (...) -> Sequence[int]
    out_shape = [0] * len(input_spatial_shape)
    if auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
        for i in range(len(input_spatial_shape)):
            out_shape[i] = int(
                np.ceil(
                    float(
                        input_spatial_shape[i])
                    / float(
                        strides_spatial[i])))
    elif auto_pad == 'VALID':
        for i in range(len(input_spatial_shape)):
            out_shape[i] = int(np.ceil(float(input_spatial_shape[i] - (kernel_spatial_shape[i] - 1)) / float(strides_spatial[i])))
    return out_shape

def pool(padded,  # type: np.ndarray
         x_shape,  # type: Sequence[int]
         kernel_shape,  # type: Sequence[int]
         strides_shape,  # type: Sequence[int]
         out_shape,  # type: Sequence[int]
         pad_shape,  # type: Sequence[int]
         pooling_type,  # type: Text
         count_include_pad=0  # type: int
         ):  # type: (...) -> np.ndarray
    spatial_size = len(x_shape) - 2
    y = np.zeros([x_shape[0], x_shape[1]] + list(out_shape))

    for shape in itertools.product(range(x_shape[0]), range(x_shape[1]), *[range(int(
            (x_shape[i + 2] + pad_shape[i] - kernel_shape[i]) / strides_shape[i] + 1)) for i in range(spatial_size)]):
        window = padded[shape[0], shape[1]]
        window_vals = np.array([window[i] for i in list(
            itertools.product(
                *[range(strides_shape[i] * shape[i + 2], strides_shape[i] * shape[i + 2] + kernel_shape[i]) for i in
                  range(spatial_size)])
        )])
        if pooling_type == 'AVG':
            f = np.average
        elif pooling_type == 'MAX':
            f = np.max
        else:
            raise NotImplementedError(
                'Pooling type {} does not support. Should be AVG, MAX'.format(pooling_type))

        if count_include_pad == 1 and pooling_type == 'AVG':
            y[shape] = f(window_vals)
        else:
            y[shape] = f(window_vals[np.where(~np.isnan(window_vals))])
    return y.astype(np.float32)

x = np.random.randn(1, 3, 32).astype(np.float32)
x_shape = np.shape(x)
kernel_shape = [2]
strides = [1]
out_shape = get_output_shape('VALID', x_shape[2:], kernel_shape, strides) # shape = [31]
padded = x
y_actual = pool(padded, x_shape, kernel_shape, strides, out_shape, [0], 'AVG') # shape = [1,3,31]


#Running the model using ONNX Runtime
import onnxruntime as rt
import numpy
sess = rt.InferenceSession("../onnx_generated_models/averagepool1.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

y_pred = sess.run(
        [label_name], {input_name: x.astype(numpy.float32)})

print("The predicted output for the operation: Conv [Without padding]")
print(y_pred)

y_pred = np.asarray(y_pred) #converting list into an array
print(y_pred.shape)

y_pred = np.squeeze(y_pred, axis=0)
print(y_pred.shape)

compare(y_actual, y_pred)