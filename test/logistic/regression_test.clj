(ns logistic.regression-test
  (:require [clojure.test :refer :all]
            [logistic.regression :refer :all]
            [logistic.mlutils :as mlu]
            [clojure.core.matrix :as m]))

(deftest simple-training-test
  (let [train-data (-> (mlu/load-data "data/spamdata/spamtrain.csv") (mlu/scale-features -1.0 1.0))
        test-data (-> (mlu/load-data "data/spamdata/spamtest.csv") (mlu/scale-features -1.0 1.0))
        params (params {:alpha 0.8 :lambda 0.0 :maxiter 1000})
        theta (fit params train-data 10)
        y-pred (predict (test-data :data) theta)
        proba (predict-proba (test-data :data) theta)
        results (mlu/compute-accuracy (test-data :labels) y-pred)]
    (is (= [1841 1] (m/shape proba)))
    (is (= 0 (m/zero-count proba)))
    (is (= [58 1] (m/shape theta)))))
