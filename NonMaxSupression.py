#NonMaxSupression [nonmaxsuppression_center_point_box_format]
import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare

# Create the inputs (ValueInfoProto)
boxes = helper.make_tensor_value_info('boxes', TensorProto.FLOAT, [1, 6, 4])
scores = helper.make_tensor_value_info('scores', TensorProto.FLOAT, [1, 1, 6])
max_output_boxes_per_class = helper.make_tensor_value_info('max_output_boxes_per_class', TensorProto.INT64, [1,])
iou_threshold = helper.make_tensor_value_info('iou_threshold', TensorProto.FLOAT, [1,])
score_threshold = helper.make_tensor_value_info('score_threshold', TensorProto.FLOAT, [1,])


# Create one output (ValueInfoProto)
selected_indices = helper.make_tensor_value_info('selected_indices', TensorProto.INT64, [3, 3])

# Create a node (NodeProto)
node_def = helper.make_node(
    'NonMaxSuppression',
    inputs=['boxes', 'scores', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold'],
    outputs=['selected_indices'],
    center_point_box=1
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold],
    [selected_indices],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-nonmaxsuppression')
print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

#model_def = onnx.utils.polish_model(model_def)
# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, 'nonmaxsuppression.onnx')
onnx.save(model_def, new_model_path)
print('The model is saved.')


# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, 'nonmaxsuppression.onnx')
onnx_model1 = onnx.load(model_path1)
print('The model is:\n{}'.format(onnx_model1))


boxes = np.array([[
    [0.5, 0.5, 1.0, 1.0],
    [0.5, 0.6, 1.0, 1.0],
    [0.5, 0.4, 1.0, 1.0],
    [0.5, 10.5, 1.0, 1.0],
    [0.5, 10.6, 1.0, 1.0],
    [0.5, 100.5, 1.0, 1.0]
]]).astype(np.float32)
scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
max_output_boxes_per_class = np.array([3]).astype(np.int64)
iou_threshold = np.array([0.5]).astype(np.float32)
score_threshold = np.array([0.0]).astype(np.float32)

selected_indices = np.array([[0, 0, 3], [0, 0, 0], [0, 0, 5]]).astype(np.int64)


#Running the model using ONNX Runtime
import onnxruntime as rt
import numpy
sess = rt.InferenceSession("nonmaxsuppression.onnx")
input_name = sess.get_inputs()[0].name
input_name1 = sess.get_inputs()[1].name
input_name2 = sess.get_inputs()[2].name
input_name3 = sess.get_inputs()[3].name
input_name4 = sess.get_inputs()[4].name
label_name = sess.get_outputs()[0].name


y_pred = sess.run(
        [label_name], {input_name: boxes.astype(numpy.float32),
        input_name1: scores.astype(numpy.float32), input_name2: max_output_boxes_per_class .astype(numpy.int64),
        input_name3: iou_threshold .astype(numpy.float32), input_name4: score_threshold .astype(numpy.float32)})

print("The predicted output for the operation: NonMaxSuppression")
print(y_pred)

y_pred = np.asarray(y_pred) #converting list into an array
print(y_pred.shape)

y_pred = np.squeeze(y_pred, axis=0)
print(y_pred.shape)

compare(selected_indices, y_pred)