# ONNX-1.6.0-OP-Library
This repository contains the python implementation of 130 + operators from ONNX operators list. This was built w.r.t ONNX 1.6.0.

Operator's List here:
https://github.com/onnx/onnx/blob/master/docs/Operators.md

WHAT you can do with the operator using this Library.

Four steps for each operator:
1. create the onnx operator and make modifications of your choice with the configurations available for that op from the onnx operators page.
2. save the ONNX model to your disk.
3. Run the operator with random input data and get the output using ONNXRuntime Framework.
4. Compare the output generated from ONNXRuntime with the actual output (numpy output)
