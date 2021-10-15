#Place the output of this code into serving/currentModel/1/assets.extra/
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow_serving.apis import model_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_log_pb2

with tf.io.TFRecordWriter("tf_serving_warmup_requests") as writer:

    data = pd.read_csv("test_data.csv", header=0, nrows=100)

    features = {name: np.array(value) for name, value in data.items()}
    features.pop("won")
    
    request = predict_pb2.PredictRequest(
        model_spec=model_pb2.ModelSpec(name="serving/currentModel/1"),
    )

    for name in features.keys():
        request.inputs[name].CopyFrom(tf.make_tensor_proto(features[name], dtype=tf.float32))
    
    log = prediction_log_pb2.PredictionLog(
        predict_log=prediction_log_pb2.PredictLog(request=request))
    writer.write(log.SerializeToString())