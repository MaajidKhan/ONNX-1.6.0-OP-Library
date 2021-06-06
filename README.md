# Build and verify ONNX Opeartors using Python.

This repository contains the python implementation of 130 + operators from ONNX operators list. This was built w.r.t ONNX 1.6.0 version.

# Documentation.
ONNX Operator's List:[https://github.com/onnx/onnx/blob/master/docs/Operators.md]

# Use
The following things can be achived by using this library of operators.

1. create the onnx operator and make modifications of your choice with the configurations available for that op from the onnx operators page.
2. save the ONNX model to your disk.
3. Run the operator with random input data and get the output using ONNXRuntime Framework [https://github.com/microsoft/onnxruntime/].
4. Compare the output generated from ONNXRuntime with the actual output (numpy output)

# Prerequisites

## Method 1: using conda environment in python

Install miniconda or conda in your machine.

```
step 1: Activate conda environment
source miniconda3/bin/activate
```

```
step 2: create virtual environment and install python >= 3.6
conda create -n onnx python=3.6
```

```
step 3: Activate the newly created venv.
To activate this environment, use

$ conda activate onnx

To deactivate an active environment, use

$ conda deactivate

```

```
step 4: verifiy python version and python installation path
$ python --version
$ which python
```

```
step 5: pip install onnx library
pip3 install onnx==1.6
```

## Method 2: using pip install
```
pip3 install requirements.txt
```

Note: After Installation, make sure the two python libraries are installed: onnx and onnxruntime

```
pip3 list
```

# Usage

```
step 1:
Make sure you are at this path:
> cd ONNX-1.6.0-OP-Library/operators

step 2: Run any operator of your choice
> python3 <op.py>

Example: python3 Abs.py

The models get saved and loaded from this directory.
ONNX-1.6.0-OP-Library/onnx_generated_models/
```