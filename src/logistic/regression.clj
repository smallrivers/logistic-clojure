(ns logistic.regression
  (:require
    [clatrix.core :as c]
    [clojure.core.matrix.operators :as mop]
    [logistic.mlutils :as mlu]))

(defn params
  [{:keys [alpha lambda maxiter gtol] :or {alpha 0.01 lambda 0.0 maxiter 100 gtol 1e-5}}]
  {:alpha   alpha
   :lambda  lambda
   :maxiter maxiter
   :gtol    gtol})

(defn- add-ones [data]
  (let [[m n] (c/size data)]
    (c/hstack (c/matrix (repeat m 1)) data)))

(defn- compute-proba [data theta]
  (let [dot (c/* data theta)
        z (c/mult dot -1)
        denom (c/+ z 1)]
    (mop/** denom -1)))

(defmacro mget
  "Faster implementation of `get`, a single value by indices only."
  ([m r] `(c/dotom .get ~m (int ~r)))
  ([m r c] `(c/dotom .get ~m (int ~r) (int ~c))))

(defn- cost-prime
  "gradient computation"
  [theta data-p]
  (let [[data y] data-p
        [m n] (c/size data)
        proba (compute-proba data theta)
        error (c/- proba y)
        grad (c/* (c/t data) error)]
    (c/div grad m)))

(defn- theta-update
  "compute theta update using gradient grad"
  [grad theta alpha lambda m]
  (let [reg (c/div (c/mult theta lambda) m)
        grad (c/+ grad reg)
        adjustment (c/mult grad alpha)]
    (c/- theta adjustment)))

(defn fit
  "Performs gradient descent to optimize the parameters for the training data.
   Note: The function is overloaded so that unecessary splitting is not processes
   if batch size parameters are missing."
  ; batch processing
  ([{:keys [alpha lambda maxiter gtol]} {:keys [data labels]}]
   (let [data (add-ones data)
         [m n] (c/size data)]
     (loop [theta (c/matrix (repeat n 0))
            grad-mag 999
            iter 0]
       (if (and (< iter maxiter) (> grad-mag gtol))
         (do (println "Iter: " iter " Mag: " grad-mag)
             (let [grad (cost-prime theta [data labels])
                   new-theta (theta-update grad theta alpha lambda m)]
               (recur new-theta (c/norm grad) (+ iter 1))))
         theta))))

  ; mini-batch processing
  ([{:keys [alpha lambda maxiter]} {:keys [data labels]} batch-size]
   (let [data (add-ones data)
         [m n] (c/size data)
         data-batches (mlu/get-mini-batches data batch-size)
         labels-batches (mlu/get-mini-batches labels batch-size)
         num-batches (count data-batches)]
     (loop [theta (c/matrix (repeat n 0))
            grad-mag 999
            iter 0]
       (if (< iter maxiter)
         (let [[new-theta grad]
               (loop [theta-batch theta
                      grad-vec []
                      i 0]
                 (if (< i num-batches)
                   (let [data (get data-batches i)
                         y (get labels-batches i)
                         [m-batch n] (c/size data)
                         grad-batch (cost-prime theta-batch [data y])
                         theta-batch (theta-update grad-batch theta-batch alpha lambda m-batch)]
                     (recur theta-batch (conj grad-vec (c/norm grad-batch)) (+ i 1)))
                   [theta-batch (mlu/avg grad-vec)]))]
           (recur new-theta grad (+ iter 1)))
         theta))))

  ; mini-batch processing with parallel gradient summation
  ([{:keys [alpha lambda maxiter]} {:keys [data labels]} batch-size num-p]
   (let [data (add-ones data)
         [m n] (c/size data)
         data-batches (mlu/get-mini-batches data batch-size)
         labels-batches (mlu/get-mini-batches labels batch-size)
         num-batches (count data-batches)]
     (loop [theta (c/matrix (repeat n 0))
            grad-mag 999
            iter 0]
       (if (< iter maxiter)
         (do (println "Iter: " iter " Mag: " grad-mag)
             (let [[new-theta grad]
                   (loop [theta-batch theta
                          grad-vec []
                          i 0]
                     (if (< i num-batches)
                       (let [data-b (get data-batches i)
                             labels-b (get labels-batches i)
                             [m-batch n-batch] (c/size data)
                             data-p (mlu/get-batches data-b num-p)
                             labels-p (mlu/get-batches labels-b num-p)
                             data-p (map vector data-p labels-p)
                             cost-prime-p (partial cost-prime theta-batch)
                             mapped (pmap cost-prime-p data-p)
                             grad-batch (reduce c/+ mapped)
                             theta-batch (theta-update grad-batch theta-batch alpha lambda m-batch)]
                         (recur theta-batch (conj grad-vec (c/norm grad-batch)) (+ i 1)))
                       [theta-batch (mlu/avg grad-vec)]))]
               (recur new-theta grad (+ iter 1))))
         theta)))))

(defn predict
  "return label predictions for data set data given parameters theta"
  [data theta]
  (let [data (add-ones data)
        proba (compute-proba data theta)]
    (c/matrix (for [pred proba]
                (if (< pred 0.5) 0 1)))))

(defn predict-proba
  "return probability predictions for data set data given parameters theta"
  [data theta]
  (let [data (add-ones data)]
    (compute-proba data theta)))

(defn print-proba
  "print each value in the probability matrix proba"
  [proba]
  (let [[m n] (c/size proba)]
    (loop [i 0]
      (if (< i m)
        (do
          (println (mget proba i 0))
          (recur (+ i 1)))))))
