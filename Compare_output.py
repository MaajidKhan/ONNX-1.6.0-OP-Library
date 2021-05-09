import numpy as np
def compare(pred, result):
  res = np.array_equal(pred, result)
  if(res):
      print("The predicted output is same as the actual reference output")
  else:
      print("Mismatch Found")