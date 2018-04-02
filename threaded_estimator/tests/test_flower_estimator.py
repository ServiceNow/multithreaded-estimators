import time

from threaded_estimator.models import FlowerClassifier, make_threaded
import numpy as np
import pytest
import tensorflow as tf


@pytest.mark.parametrize('fe',[FlowerClassifier(), make_threaded(FlowerClassifier())])
def test_iris_estimator_trains(fe):
    fe.train(steps=10)

predict_x = {
    'SepalLength': [5.1],
    'SepalWidth': [3.3],
    'PetalLength': [1.7],
    'PetalWidth': [0.5],
}


def test_normal_input_fn():
    fe = FlowerClassifier()
    ds = fe.predict_input_fn(predict_x)
    value = ds.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
        features = sess.run(value)

    assert isinstance(features, dict)


def test_prediction_api_is_constant():

    fe_threaded = make_threaded(FlowerClassifier())
    fe_unthreaded = FlowerClassifier()

    p1 = fe_threaded.predict(predict_x)
    p2 = fe_unthreaded.predict(predict_x)

    assert type(p1) is type(p2)
    assert len(p1) is len(p2)


def test_predictions_change_with_training():

    fe = FlowerClassifier()
    predictions1 = fe.predict(features=predict_x)
    fe.train(steps=100)
    predictions2 = fe.predict(features=predict_x)

    print(predictions1['logits'])
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(predictions1['logits'],
                                      predictions2['logits'])


@pytest.mark.parametrize('fe',[FlowerClassifier(), make_threaded(FlowerClassifier())])
def test_iris_estimator_predict_deterministic(fe):

    predictions1 = fe.predict(features=predict_x)
    predictions2 = fe.predict(features=predict_x)

    print(predictions1)
    print(predictions2)

    np.testing.assert_array_equal(predictions1['logits'],
                                  predictions2['logits'])


def test_threaded_faster_than_non_threaded():

    fe_threaded = make_threaded(FlowerClassifier())
    fe_unthreaded = FlowerClassifier()

    n_epochs = 100

    print('starting unthreaded')
    t1 = time.time()
    for _ in range(n_epochs):
        predictions = fe_unthreaded.predict(features=predict_x)

    print('starting threaded')
    t2 = time.time()
    for _ in range(n_epochs):
        predictions = fe_threaded.predict(features=predict_x)

    t3 = time.time()

    unthreaded_time = (t2-t1)
    threaded_time = (t3-t2)

    assert unthreaded_time > threaded_time

    print(f'Threaded time was {threaded_time}; s\n'
          f'Unthreaded time was {unthreaded_time};  s\n'
          f'Threaded was {unthreaded_time/threaded_time} times faster!')




