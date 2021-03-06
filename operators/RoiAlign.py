#RoiAlign
import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare

# Create one Input (ValueInfoProto)
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 1, 10, 10])
rois = helper.make_tensor_value_info('rois', TensorProto.FLOAT, [3, 4])
batch_indices = helper.make_tensor_value_info('batch_indices', TensorProto.INT64, [3,])

# Create one output (ValueInfoProto)
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [3, 1, 5, 5])

# Create a node (NodeProto)
node_def = helper.make_node(
    "RoiAlign",
    inputs=["X", "rois", "batch_indices"],
    outputs=["Y"],
    spatial_scale=1.0,
    output_height=5,
    output_width=5,
    sampling_ratio=2,
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [X, rois, batch_indices],
    [Y],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-RoiAlign')

print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, '../onnx_generated_models/onnx-RoiAlign.onnx')
onnx.save(model_def, new_model_path)

print('The model is saved.')



# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, '../onnx_generated_models/onnx-RoiAlign.onnx')
onnx_model1 = onnx.load(model_path1)

print('The model is:\n{}'.format(onnx_model1))

import onnxruntime as rt
sess = rt.InferenceSession("../onnx_generated_models/onnx-RoiAlign.onnx")
input_name = sess.get_inputs()[0].name
input_name1 = sess.get_inputs()[1].name
input_name2 = sess.get_inputs()[2].name
label_name = sess.get_outputs()[0].name

X = np.array(
    [
        [
            [
                [
                    0.2764,
                    0.7150,
                    0.1958,
                    0.3416,
                    0.4638,
                    0.0259,
                    0.2963,
                    0.6518,
                    0.4856,
                    0.7250,
                ],
                [
                    0.9637,
                    0.0895,
                    0.2919,
                    0.6753,
                    0.0234,
                    0.6132,
                    0.8085,
                    0.5324,
                    0.8992,
                    0.4467,
                ],
                [
                    0.3265,
                    0.8479,
                    0.9698,
                    0.2471,
                    0.9336,
                    0.1878,
                    0.4766,
                    0.4308,
                    0.3400,
                    0.2162,
                ],
                [
                    0.0206,
                    0.1720,
                    0.2155,
                    0.4394,
                    0.0653,
                    0.3406,
                    0.7724,
                    0.3921,
                    0.2541,
                    0.5799,
                ],
                [
                    0.4062,
                    0.2194,
                    0.4473,
                    0.4687,
                    0.7109,
                    0.9327,
                    0.9815,
                    0.6320,
                    0.1728,
                    0.6119,
                ],
                [
                    0.3097,
                    0.1283,
                    0.4984,
                    0.5068,
                    0.4279,
                    0.0173,
                    0.4388,
                    0.0430,
                    0.4671,
                    0.7119,
                ],
                [
                    0.1011,
                    0.8477,
                    0.4726,
                    0.1777,
                    0.9923,
                    0.4042,
                    0.1869,
                    0.7795,
                    0.9946,
                    0.9689,
                ],
                [
                    0.1366,
                    0.3671,
                    0.7011,
                    0.6234,
                    0.9867,
                    0.5585,
                    0.6985,
                    0.5609,
                    0.8788,
                    0.9928,
                ],
                [
                    0.5697,
                    0.8511,
                    0.6711,
                    0.9406,
                    0.8751,
                    0.7496,
                    0.1650,
                    0.1049,
                    0.1559,
                    0.2514,
                ],
                [
                    0.7012,
                    0.4056,
                    0.7879,
                    0.3461,
                    0.0415,
                    0.2998,
                    0.5094,
                    0.3727,
                    0.5482,
                    0.0502,
                ],
            ]
        ]
    ],
    dtype=np.float32,
)
batch_indices = np.array([0, 0, 0], dtype=np.int64)
rois = np.array([[0, 0, 9, 9], [0, 5, 4, 9], [5, 5, 9, 9]], dtype=np.float32)
# (num_rois, C, output_height, output_width)
y_actual = np.array(
    [
        [
            [
                [0.4664, 0.4466, 0.3405, 0.5688, 0.6068],
                [0.3714, 0.4296, 0.3835, 0.5562, 0.3510],
                [0.2768, 0.4883, 0.5222, 0.5528, 0.4171],
                [0.4713, 0.4844, 0.6904, 0.4920, 0.8774],
                [0.6239, 0.7125, 0.6289, 0.3355, 0.3495],
            ]
        ],
        [
            [
                [0.3022, 0.4305, 0.4696, 0.3978, 0.5423],
                [0.3656, 0.7050, 0.5165, 0.3172, 0.7015],
                [0.2912, 0.5059, 0.6476, 0.6235, 0.8299],
                [0.5916, 0.7389, 0.7048, 0.8372, 0.8893],
                [0.6227, 0.6153, 0.7097, 0.6154, 0.4585],
            ]
        ],
        [
            [
                [0.2384, 0.3379, 0.3717, 0.6100, 0.7601],
                [0.3767, 0.3785, 0.7147, 0.9243, 0.9727],
                [0.5749, 0.5826, 0.5709, 0.7619, 0.8770],
                [0.5355, 0.2566, 0.2141, 0.2796, 0.3600],
                [0.4365, 0.3504, 0.2887, 0.3661, 0.2349],
            ]
        ],
    ],
    dtype=np.float32,
)

y_pred = sess.run(
        [label_name], {input_name: X, input_name1: rois,
        input_name2: batch_indices })

print("The predicted output for the operation: RoiAlign", y_pred)

y_pred = np.asarray(y_pred) #converting list into an array
print(y_pred.shape)

y_pred = np.squeeze(y_pred, axis=0)
print(y_pred.shape)

y_pred1 = np.round(y_pred, 2)
y_actual1 = np.round(y_actual, 2)
compare(y_pred1, y_actual1)