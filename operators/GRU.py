#GRU [default]
import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare

# Create the inputs (ValueInfoProto)
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 3, 2])
W = helper.make_tensor_value_info('W', TensorProto.FLOAT, [1, 15, 2])
R = helper.make_tensor_value_info('R', TensorProto.FLOAT, [1, 15, 5])


# Create one output (ValueInfoProto)
y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 3, 5])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 3, 5])

hidden_size = 5
# Create a node (NodeProto)
node_def = helper.make_node(
    'GRU',
    inputs=['X', 'W', 'R'],
    outputs=['y', 'Y'],
    hidden_size=hidden_size
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [X,W,R],
    [y,Y]
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-gru')
print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

#model_def = onnx.utils.polish_model(model_def)
# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, '../onnx_generated_models/gru.onnx')
onnx.save(model_def, new_model_path)
print('The model is saved.')


# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, '../onnx_generated_models/gru.onnx')
onnx_model1 = onnx.load(model_path1)
print('The model is:\n{}'.format(onnx_model1))

input = np.array([[[1., 2.], [3., 4.], [5., 6.]]]).astype(np.float32)

input_size = 2
hidden_size = 5
weight_scale = 0.1
number_of_gates = 3

class GRU_Helper():
    def __init__(self, **params):  # type: (*Any) -> None
        # GRU Input Names
        X = str('X')
        W = str('W')
        R = str('R')
        B = str('B')
        H_0 = str('initial_h')
        LBR = str('linear_before_reset')
        number_of_gates = 3

        required_inputs = [X, W, R]
        for i in required_inputs:
            assert i in params, "Missing Required Input: {0}".format(i)

        self.num_directions = params[W].shape[0]

        if self.num_directions == 1:
            for k in params.keys():
                if k != X:
                    params[k] = np.squeeze(params[k], axis=0)

            hidden_size = params[R].shape[-1]
            batch_size = params[X].shape[1]

            b = params[B] if B in params else np.zeros(2 * number_of_gates * hidden_size)
            h_0 = params[H_0] if H_0 in params else np.zeros((batch_size, hidden_size))
            lbr = params[LBR] if LBR in params else 0

            self.X = params[X]
            self.W = params[W]
            self.R = params[R]
            self.B = b
            self.H_0 = h_0
            self.LBR = lbr

        else:
            raise NotImplementedError()

    def f(self, x):  # type: (np.ndarray) -> np.ndarray
        return 1 / (1 + np.exp(-x))

    def g(self, x):  # type: (np.ndarray) -> np.ndarray
        return np.tanh(x)

    def step(self):  # type: () -> Tuple[np.ndarray, np.ndarray]
        h_list = []
        [w_z, w_r, w_h] = np.split(self.W, 3)
        [r_z, r_r, r_h] = np.split(self.R, 3)
        [w_bz, w_br, w_bh, r_bz, r_br, r_bh] = np.split(self.B, 6)
        gates_w = np.transpose(np.concatenate((w_z, w_r)))
        gates_r = np.transpose(np.concatenate((r_z, r_r)))
        gates_b = np.add(np.concatenate((w_bz, w_br)), np.concatenate((r_bz, r_br)))

        H_t = self.H_0
        for x in np.split(self.X, self.X.shape[0], axis=0):
            gates = np.dot(x, gates_w) + np.dot(H_t, gates_r) + gates_b
            z, r = np.split(gates, 2, -1)
            z = self.f(z)
            r = self.f(r)
            h_default = self.g(np.dot(x, np.transpose(w_h)) + np.dot(r * H_t, np.transpose(r_h)) + w_bh + r_bh)
            h_linear = self.g(np.dot(x, np.transpose(w_h)) + r * (np.dot(H_t, np.transpose(r_h)) + r_bh) + w_bh)
            h = h_linear if self.LBR else h_default
            H = (1 - z) * h + z * H_t
            h_list.append(H)
            H_t = H
        concatenated = np.concatenate(h_list)
        if self.num_directions == 1:
            output = np.expand_dims(concatenated, 1)
        return output, h_list[-1]


W = weight_scale * np.ones((1, number_of_gates * hidden_size, input_size)).astype(np.float32)
R = weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32)

gru = GRU_Helper(X=input, W=W, R=R)
_, Y_h = gru.step()

Y_h = np.array(Y_h).astype(np.float32)
#Running the model using ONNX Runtime
import onnxruntime as rt
import numpy
sess = rt.InferenceSession("../onnx_generated_models/gru.onnx")
input_name = sess.get_inputs()[0].name
input_name1 = sess.get_inputs()[1].name
input_name2 = sess.get_inputs()[2].name
label_name = sess.get_outputs()[0].name
label_name1 = sess.get_outputs()[1].name


y_pred = sess.run(
        [], {input_name: input.astype(numpy.float32),
        input_name1: W.astype(numpy.float32), input_name2: R.astype(numpy.float32)})

print("The predicted output for the operation: GRU")
print(y_pred[1])
y_pred[1] = np.round(y_pred[1], 6)
Y_h = np.round(Y_h, 6)
compare(Y_h, y_pred[1])