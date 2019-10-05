trace
===========================

.. automodule:: fastestimator.trace
   :members:
   :undoc-members:
   :show-inheritance:

.. toctree::
   :maxdepth: 1
   
   fastestimator.trace.adapt
   fastestimator.trace.io
   fastestimator.trace.metric

----------------------------------

trace
----------------------------------
.. autoclass:: fastestimator.trace.trace.Trace
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: fastestimator.trace.trace.TrainInfo
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: on_batch_begin, on_batch_end, on_begin, on_end, on_epoch_begin, on_epoch_end

.. autoclass:: fastestimator.trace.trace.MonitorLoss
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: on_batch_begin, on_batch_end, on_begin, on_end, on_epoch_begin, on_epoch_end
