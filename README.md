logistic-clojure
================

A vectorized implementation of a logistic regression classifier written in Clojure using clatrix.core 0.3.0.

This library supports mini-batch processing and parallel gradient summation.  The -main function in core.clj provides a simple test of the algorithm on the SPAMBase data set.  Alter the last two parameters of the "lr/fit" call (core.clj ln:73) to adjust the mini-batch size and the level of parallelism.
