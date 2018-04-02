from threaded_estimator.models import FlowerClassifier
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
fc = FlowerClassifier(model_path='../trained_models')

fc.train(steps=1000)

predict_x = {
    'SepalLength': [5.1],
    'SepalWidth': [3.3],
    'PetalLength': [1.7],
    'PetalWidth': [0.5],
}

p1 = list(fc.predict(predict_x))
# INFO:tensorflow:Restoring parameters from ./trained_models/model.ckpt-5000

p2 = list(fc.predict(predict_x))
# INFO:tensorflow:Restoring parameters from ./trained_models/model.ckpt-5000


