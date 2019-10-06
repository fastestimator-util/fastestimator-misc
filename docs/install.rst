============
Installation
============

Prerequisites
-------------
- Python3
- TensorFlow2

   * GPU:  ``pip install tensorflow-gpu==2.0.0``
   * CPU:  ``pip install tensorflow==2.0.0``

Installation
----------------
``pip install fastestimator==1.0b0``

Docker
-------

Docker container creates isolated virtual environment that shares resources with host machine.
Docker provides an easy way to set up FastEstimator running environment, users can either build
image from dockerfile or pull image from Docker-Hub_.

- GPU:  ``docker pull fastestimator/fastestimator:1.0b0-gpu``
- CPU:  ``docker pull fastestimator/fastestimator:1.0b0-cpu``

.. _Docker-Hub: https://hub.docker.com/r/fastestimator/fastestimator/tags