import time
import tensorflow as tf
from threaded_estimator.models import FlowerClassifier, FlowerClassifierThreaded


tf.logging.set_verbosity(tf.logging.INFO)

predict_x = {
    'SepalLength': [5.1],
    'SepalWidth': [3.3],
    'PetalLength': [1.7],
    'PetalWidth': [0.5],
}

fe_threaded = FlowerClassifierThreaded(threaded=True)
fe_unthreaded = FlowerClassifier()

n_epochs = 100

print('starting unthreaded')
t1 = time.time()
for _ in range(n_epochs):
    predictions = list(fe_unthreaded.predict(features=predict_x))

print('starting threaded')
t2 = time.time()
for _ in range(n_epochs):
    predictions = list(fe_threaded.predict(features=predict_x))

t3 = time.time()

unthreaded_time = (t2 - t1)
threaded_time = (t3 - t2)

assert unthreaded_time > threaded_time

print(f'Threaded time was {threaded_time}s;\n'
      f'Unthreaded time was {unthreaded_time}s; \n'
      f'Threaded was {unthreaded_time/threaded_time} times faster!')
