import time

from threaded_estimator import models
import numpy as np
import pytest
import tensorflow as tf


def test_iris_estimator_trains():
    fe = models.FlowerClassifierThreaded(threaded=False)
    fe.train(steps=10)

predict_x = {
    'SepalLength': [5.1],
    'SepalWidth': [3.3],
    'PetalLength': [1.7],
    'PetalWidth': [0.5],
}


def test_normal_input_fn():
    fe = models.FlowerClassifierThreaded(threaded=False)
    ds = fe.predict_input_fn(predict_x)
    value = ds.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
        features = sess.run(value)

    assert isinstance(features, dict)


def test_predictions_change_with_training():
    fe = models.FlowerClassifierThreaded(threaded=False)
    predictions1 = list(fe.predict(features=predict_x))
    fe.train(steps=100)
    predictions2 = list(fe.predict(features=predict_x))

    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(predictions1[0]['logits'],
                                      predictions2[0]['logits'])


@pytest.mark.parametrize('threaded', [False, True])
def test_iris_estimator_predict_deterministic(threaded):
    fe = models.FlowerClassifierThreaded(threaded=threaded)
    predictions1 = fe.predict(features=predict_x)
    predictions2 = fe.predict(features=predict_x)

    if not threaded:
        predictions1 = list(predictions1)[0]
        predictions2 = list(predictions2)[0]

    print(threaded, predictions1)
    print(threaded, predictions2)
    np.testing.assert_array_equal(predictions1['logits'],
                                  predictions2['logits'])




def test_threaded_faster_than_non_threaded():

    fe_threaded = models.FlowerClassifierThreaded(threaded=True)
    fe_threaded.train(1000)
    fe_unthreaded = models.FlowerClassifierThreaded(threaded=False)
    fe_unthreaded.train(1000)

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

    unthreaded_time = (t2-t1)
    threaded_time = (t3-t2)

    assert unthreaded_time > threaded_time

    print(f'Threaded time was {threaded_time};  '
          f'Unthreaded time was {unthreaded_time};  '
          f'Threaded was {unthreaded_time/threaded_time} times faster!')




