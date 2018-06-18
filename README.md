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

This code is released under Apache 2. See the [license in full](LICENSE).
