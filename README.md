*ServiceNow completed its acquisition of Element AI on January 8, 2021. All references to Element AI in the materials that are part of this project should refer to ServiceNow.*

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

This code is released under an Apache 2 license. See [the license in full](LICENSE).
