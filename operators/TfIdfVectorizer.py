#TfIdfVectorizer (tf_batch_onlybigrams_skip0)

import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
from Compare_output import compare

# Create the inputs (ValueInfoProto)
x = helper.make_tensor_value_info('x', TensorProto.INT32, [2, 6])


# Create one output (ValueInfoProto)
y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [2, 7])

ngram_counts = np.array([0, 4]).astype(np.int64)
ngram_indexes = np.array([0, 1, 2, 3, 4, 5, 6]).astype(np.int64)
pool_int64s = np.array([2, 3, 5, 4,    # unigrams
                        5, 6, 7, 8, 6, 7]).astype(np.int64)   # bigrams

class TfIdfVectorizerHelper():
    def __init__(self, **params):    # type: (*Any) -> None
        # Attr names
        mode = str('mode')
        min_gram_length = str('min_gram_length')
        max_gram_length = str('max_gram_length')
        max_skip_count = str('max_skip_count')
        ngram_counts = str('ngram_counts')
        ngram_indexes = str('ngram_indexes')
        pool_int64s = str('pool_int64s')

        required_attr = [mode, min_gram_length, max_gram_length, max_skip_count,
                         ngram_counts, ngram_indexes, pool_int64s]

        for i in required_attr:
            assert i in params, "Missing attribute: {0}".format(i)

        self.mode = params[mode]
        self.min_gram_length = params[min_gram_length]
        self.max_gram_length = params[max_gram_length]
        self.max_skip_count = params[max_skip_count]
        self.ngram_counts = params[ngram_counts]
        self.ngram_indexes = params[ngram_indexes]
        self.pool_int64s = params[pool_int64s]

    def make_node_noweights(self):    # type: () -> NodeProto
        return onnx.helper.make_node(
            'TfIdfVectorizer',
            inputs=['x'],
            outputs=['y'],
            mode=self.mode,
            min_gram_length=self.min_gram_length,
            max_gram_length=self.max_gram_length,
            max_skip_count=self.max_skip_count,
            ngram_counts=self.ngram_counts,
            ngram_indexes=self.ngram_indexes,
            pool_int64s=self.pool_int64s
        )


# Create a node (NodeProto)
helper = TfIdfVectorizerHelper(
    mode='TF',
    min_gram_length=2,
    max_gram_length=2,
    max_skip_count=0,
    ngram_counts=ngram_counts,
    ngram_indexes=ngram_indexes,
    pool_int64s=pool_int64s
)

node_def = helper.make_node_noweights()

from onnx import helper
# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [x],
    [y],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-TfIdfVectorizer')
print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

# Save the ONNX model
import os
path = os.getcwd()
new_model_path = os.path.join(path, '../onnx_generated_models/TfIdfVectorizer.onnx')
onnx.save(model_def, new_model_path)
print('The model is saved.')


# Preprocessing: load the ONNX model (Loading an already exisisting model)
model_path1 = os.path.join(path, '../onnx_generated_models/TfIdfVectorizer.onnx')
onnx_model1 = onnx.load(model_path1)
print('The model is:\n{}'.format(onnx_model1))


x = np.array([[1, 1, 3, 3, 3, 7], [8, 6, 7, 5, 6, 8]]).astype(np.int32)
y_actual = np.array([[0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 1., 0., 1.]]).astype(np.float32)



#Running the model using ONNX Runtime
import onnxruntime as rt
import numpy
sess = rt.InferenceSession("../onnx_generated_models/TfIdfVectorizer.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

y_pred = sess.run(
        [label_name], {input_name: x.astype(numpy.int32),
        })

print("The predicted output for the operation: TfIdfVectorizer")
print(y_pred)

y_pred = np.asarray(y_pred) #converting list into an array
print(y_pred.shape)

y_pred = np.squeeze(y_pred, axis=0)
print(y_pred.shape)

compare(y_actual, y_pred)