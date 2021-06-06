#LSTM [default]
import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare

# Create the inputs (ValueInfoProto)
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 3, 2])
W = helper.make_tensor_value_info('W', TensorProto.FLOAT, [1, 12, 2])
R = helper.make_tensor_value_info('R', TensorProto.FLOAT, [1, 12, 3])

# Create one output (ValueInfoProto)
y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 3, 3])
y_h = helper.make_tensor_value_info('y_h', TensorProto.FLOAT, [1, 3, 3])

hidden_size = 3
# Create a node (NodeProto)
node_def = helper.make_node(
    'LSTM',
    inputs=['X', 'W', 'R'],
    outputs=['y', 'y_h'],
    hidden_size=hidden_size
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [X, W, R],
    [y, y_h],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-lstm')
print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

#model_def = onnx.utils.polish_model(model_def)
# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, '../onnx_generated_models/lstm.onnx')
onnx.save(model_def, new_model_path)
print('The model is saved.')


# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, '../onnx_generated_models/lstm.onnx')
onnx_model1 = onnx.load(model_path1)
print('The model is:\n{}'.format(onnx_model1))


input = np.array([[[1., 2.], [3., 4.], [5., 6.]]]).astype(np.float32)

input_size = 2
hidden_size = 3
weight_scale = 0.1
number_of_gates = 4

class LSTM_Helper():
    def __init__(self, **params):  # type: (*Any) -> None
        # LSTM Input Names
        X = str('X')
        W = str('W')
        R = str('R')
        B = str('B')
        H_0 = str('initial_h')
        C_0 = str('initial_c')
        P = str('P')
        number_of_gates = 4
        number_of_peepholes = 3

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

            b = params[B] if B in params else np.zeros(2 * number_of_gates * hidden_size, dtype=np.float32)
            p = params[P] if P in params else np.zeros(number_of_peepholes * hidden_size, dtype=np.float32)
            h_0 = params[H_0] if H_0 in params else np.zeros((batch_size, hidden_size), dtype=np.float32)
            c_0 = params[C_0] if C_0 in params else np.zeros((batch_size, hidden_size), dtype=np.float32)

            self.X = params[X]
            self.W = params[W]
            self.R = params[R]
            self.B = b
            self.P = p
            self.H_0 = h_0
            self.C_0 = c_0
        else:
            raise NotImplementedError()

    def f(self, x):  # type: (np.ndarray) -> np.ndarray
        return 1 / (1 + np.exp(-x))

    def g(self, x):  # type: (np.ndarray) -> np.ndarray
        return np.tanh(x)

    def h(self, x):  # type: (np.ndarray) -> np.ndarray
        return np.tanh(x)

    def step(self):  # type: () -> Tuple[np.ndarray, np.ndarray]
        [p_i, p_o, p_f] = np.split(self.P, 3)
        h_list = []
        H_t = self.H_0
        C_t = self.C_0
        for x in np.split(self.X, self.X.shape[0], axis=0):
            gates = np.dot(x, np.transpose(self.W)) + np.dot(H_t, np.transpose(self.R)) + np.add(
                *np.split(self.B, 2))
            i, o, f, c = np.split(gates, 4, -1)
            i = self.f(i + p_i * C_t)
            f = self.f(f + p_f * C_t)
            c = self.g(c)
            C = f * C_t + i * c
            o = self.f(o + p_o * C)
            H = o * self.h(C)
            h_list.append(H)
            H_t = H
            C_t = C
        concatenated = np.concatenate(h_list)
        if self.num_directions == 1:
            output = np.expand_dims(concatenated, 1)
        return output, h_list[-1]

W = weight_scale * np.ones((1, number_of_gates * hidden_size, input_size)).astype(np.float32)
R = weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32)

lstm = LSTM_Helper(X=input, W=W, R=R)
_, Y_h = lstm.step()



#Running the model using ONNX Runtime
import onnxruntime as rt
import numpy
sess = rt.InferenceSession("../onnx_generated_models/lstm.onnx")
input_name = sess.get_inputs()[0].name
input_name1 = sess.get_inputs()[1].name
input_name2 = sess.get_inputs()[2].name
label_name = sess.get_outputs()[0].name
label_name1 = sess.get_outputs()[1].name


y_pred = sess.run(
        [], {input_name: input.astype(numpy.float32),
        input_name1: W.astype(numpy.float32), input_name2: R.astype(numpy.float32)})

print("The predicted output for the operation: LSTM")
print(y_pred)
print(y_pred[1])

y_pred[1] = np.asarray(y_pred[1]) #converting list into an array
print(y_pred[1].shape)

y_pred[1] = np.round(y_pred[1], 6)
Y_h = np.round(Y_h, 6)

compare(Y_h, y_pred[1])