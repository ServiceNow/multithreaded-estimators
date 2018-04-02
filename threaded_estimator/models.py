""" Module to expose trained models for inference."""

import tensorflow as tf

from tensorflow.contrib.learn import RunConfig
from queue import Queue
from threading import Thread

from threaded_estimator import iris_data


class FlowerClassifier:
    """ A light wrapper to handle training and inference with the Iris classifier here:
     https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/learn/iris.py

    Based upon a simple implementation of the canned TF DNNClassifier Estimator, see the tutorial here:
    https://www.tensorflow.org/versions/r1.3/get_started/estimator#construct_a_deep_neural_network_classifier

    """

    def __init__(self, model_path='./trained_models/',
                 verbose=False):
        """
        Parameters
        ----------
        model_path: str
            Location from which to load the model.

        verbose: Bool
            Whether to print various messages.
        """

        (self.train_x, self.train_y), (self.test_x, self.test_y) = iris_data.load_data()
        self.batch_size = 32

        self.model_path = model_path

        self.estimator = self.load_estimator()

        self.verbose = verbose

    def predict(self, features):
        """
        Vanilla prediction function. Returns a generator

        Intended for single-shot usage.

        Parameters
        ----------
        features: dict
            dict of input features, containing keys 'SepalLength'
                                                    'SepalWidth'
                                                    'PetalLength'
                                                    'PetalWidth'

        Returns
        -------
        predictions: generator
            Yields dictionaries containing  'probs'
                                            'outputs'
                                            'predicted_class'

        """

        return self.estimator.predict(input_fn=lambda: self.predict_input_fn(features))

    def load_estimator(self):
        """

        Returns
        -------
        estimator
            A tf.estimator.DNNClassifier

        """

        # Feature columns describe how to use the input.
        my_feature_columns = []
        for key in self.train_x.keys():
            my_feature_columns.append(tf.feature_column.numeric_column(key=key))

        run_config = RunConfig()
        run_config = run_config.replace(model_dir=self.model_path)

        return tf.estimator.DNNClassifier(
            feature_columns=my_feature_columns,
            # Two hidden layers of 10 nodes each.
            hidden_units=[10, 10],
            # The model must choose between 3 classes.
            n_classes=3,
            # Use runconfig to load model,
            config=run_config,
            model_dir=self.model_path)

    def train(self, steps):
        """
        Parameters
        ----------
        steps: int
            Number of steps to train for.

        """
        self.estimator.train(
            input_fn=lambda: self.train_input_fn(self.train_x, self.train_y),
            steps=steps,
        )

    def train_input_fn(self, features, labels):
        """
        For background on the data, see iris_data.py

        Parameters
        ----------
        features: pandas dataframe
            With columns  'SepalLength'
                          'SepalWidth'
                          'PetalLength'
                          'PetalWidth'

        labels: array
            Flower names

        Returns
        -------
        dataset: generator
            Yields batches of size self.batch_size

        """

        # Convert the inputs to a Dataset.
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

        # Shuffle, repeat, and batch the examples.
        dataset = dataset.shuffle(1000).repeat().batch(self.batch_size)

        # Return the dataset.
        return dataset

    def predict_input_fn(self, features):
        """

        Parameters
        ----------
        features:  pandas dataframe or dict
            with columns or keys  'SepalLength'
                                  'SepalWidth'
                                  'PetalLength'
                                  'PetalWidth'

        Returns
        -------
        dataset: generator
            Yields batches of size self.batch_size

        """

        if self.verbose:
            print("Standard predict_input_n call")

        # Convert the inputs to a Dataset.
        dataset = tf.data.Dataset.from_tensor_slices(dict(features))

        # Batch the examples
        assert self.batch_size is not None, "batch_size must not be None"
        dataset = dataset.batch(self.batch_size)

        # Return the batched dataset. This is an iterator returning batches.
        return dataset


class ThreadedPrediction:

    # Decorator to render the basic flower estimator threaded and fast

    def __init__(self, f):
        """
        Parameters
        ----------
        f: the function being decorated

        """

        self.f = f

        self.input_queue = Queue(maxsize=1)
        self.output_queue = Queue(maxsize=1)

        self.threaded = True

        if self.threaded:
            # We set the generator thread as daemon
            # (see https://docs.python.org/3/library/threading.html#threading.Thread.daemon)
            # This means that when all other threads are dead,
            # this thread will not prevent the Python program from exiting
            self.prediction_thread = Thread(target=self.predict_from_queue, daemon=True)
            self.prediction_thread.start()

    def __getattr__(self, k):
        try:
            return getattr(self.f, k)
        except KeyError:
            raise AttributeError

    def generate_from_queue(self):
        """ Generator which yields items from the input queue.
        This lives within our 'prediction thread'.

        """

        while True:
            if self.f.verbose:
                print('Yielding from input queue')
            yield self.input_queue.get()

    def predict_from_queue(self):
        """ Adds a prediction from the model to the output_queue.

        This lives within our 'prediction thread'.

        Note: estimators accept generators as inputs and return generators as output. Here, we are
        iterating through the output generator, which will be populated in lock-step with the input
        generator.

        """

        for i in self.f.estimator.predict(input_fn=self.queued_predict_input_fn):
            if self.f.verbose:
                print('Putting in output queue')
            self.output_queue.put(i)

    @staticmethod
    def predict_gen(predictions):

        for p in [predictions]:
            yield p

    def predict(self, features):
        """
        Overwrites .predict in FlowerClassifierBasic.

        Calls either the vanilla or multi-threaded prediction methods based upon self.threaded.

        Parameters
        ----------
        features: dict
            dict of input features, containing keys 'SepalLength'
                                                    'SepalWidth'
                                                    'PetalLength'
                                                    'PetalWidth'

        Returns
        -------
        predictions: dict
            Dictionary containing   'probs'
                                    'outputs'
                                    'predicted_class'

        """

        # Get predictions dictionary

        features = dict(features)
        self.input_queue.put(features)
        predictions = self.output_queue.get()  # The latest predictions dict
        predictions = self.predict_gen(predictions) # Yield a generator, to match API of base class

        return predictions

    def queued_predict_input_fn(self):
        """
        Queued version of the `predict_input_fn` in FlowerClassifier.

        Instead of yielding a dataset from data as a parameter, we construct a Dataset from a generator,
        which yields from the input queue.

        """

        if self.f.verbose:
            print("QUEUED INPUT FUNCTION CALLED")

        # Fetch the inputs from the input queue
        dataset = tf.data.Dataset.from_generator(self.generate_from_queue,
                                                 output_types={'SepalLength': tf.float32,
                                                               'SepalWidth': tf.float32,
                                                               'PetalLength': tf.float32,
                                                               'PetalWidth': tf.float32})

        return dataset
