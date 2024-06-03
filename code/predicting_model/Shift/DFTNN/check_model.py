import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn

path = "/home/s3665828/Documents/Masters_Thesis/repo/CASCADE/code/predicting_model/Shift/DFTNN/"
model = tf.saved_model.load(path + "gnn/models/")
signature_fn = model.signatures[
    tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
print(signature_fn)
input_dict = {"examples": ...}
output_dict = signature_fn(**input_dict)
logits = output_dict["logits"]

'''print(signature_fn)
print(input_dict)
print(output_dict)
print(logits)'''