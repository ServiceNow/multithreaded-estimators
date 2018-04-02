# Multithreaded-estimators

Code demonstrating how to use multithreading to speedup inference for Tensorflow estimators.

## Installation

A Dockerfile is provided. First build the image from the root directory:

```
docker build . -t threaded
```

Then run the tests:

```
docker run threaded
```

## License

This code needs a license. See issue https://github.com/ElementAI/multithreaded-estimators/issues/1