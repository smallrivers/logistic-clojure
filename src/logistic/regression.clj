(ns logistic.regression

  (:require
    [clatrix.core :as c]
    [clojure.core.matrix :as m]
    [clojure.core.matrix.operators :as mop]

    [logistic.mlutils :as mlu]))


; ------------------------
; Parameters

; default paramters
(def def-params {
                 :alpha   0.01
                 :lambda  0.0
                 :maxiter 100
                 :gtol    1e-5
                 })

; parameter customization (fills missing paramters with default values)
(defn get-params
  ([] def-params)
  ([alpha] {
            :alpha   alpha
            :lambda  (def-params :lambda)
            :maxiter (def-params :maxiter)
            :gtol    (def-params :gtol)
            })
  ([alpha lambda] {
                   :alpha   alpha
                   :lambda  lambda
                   :maxiter (def-params :maxiter)
                   :gtol    (def-params :gtol)
                   })
  ([alpha lambda maxiter] {
                           :alpha   alpha
                           :lambda  lambda
                           :maxiter maxiter
                           :gtol    (def-params :gtol)
                           })
  ([alpha lambda maxiter gtol] {
                                :alpha   alpha
                                :lambda  lambda
                                :maxiter maxiter
                                :gtol    gtol
                                }))


; ---------------------------
; Utilities

(defn- unpack-params [params]
  (list (params :alpha) (params :lambda) (params :maxiter) (params :gtol)))

(defn- unpack-data [data]
  (list (data :data) (data :labels)))

(defn- add-ones [X]
  (let [[m n] (c/size X)]
    (c/hstack (c/matrix (repeat m 1)) X)))

(defn- compute-proba [X theta]
  (let [dot (c/* X theta)
        z (c/mult dot -1)
        tmp (c/exp! z)
        denom (c/+ z 1)]
    (mop/** denom -1)))


;-----------------------------
; Optimization

(defmacro mget
  "Faster implementation of `get`, a single value by indices only."
  ([m r] `(c/dotom .get ~m (int ~r)))
  ([m r c] `(c/dotom .get ~m (int ~r) (int ~c))))

; gradient computation
(defn- cost-prime [theta data-p]
  (let [[X y] data-p
        [m n] (c/size X)
        proba (compute-proba X theta)
        error (c/- proba y)
        grad (c/* (c/t X) error)]
    (c/div grad m)))

; compute theta update using gradient grad
(defn- theta-update [grad theta alpha lambda m]
  (let [bias (mget grad 0 0)
        reg (c/div (c/mult theta lambda) m)
        grad (c/+ grad reg)
        tmp (c/set grad 0 0 bias)
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
  ([params data batch-size]
   (let [[alpha lambda maxiter gtol] (unpack-params params)
         [X y] (unpack-data data)
         X (add-ones X)
         [m n] (c/size X)
         X-batches (mlu/get-mini-batches X batch-size)
         y-batches (mlu/get-mini-batches y batch-size)
         num-batches (count X-batches)]
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
                       (let [X (get X-batches i)
                             y (get y-batches i)
                             [m-batch n] (c/size X)
                             grad-batch (cost-prime theta-batch [X y])
                             theta-batch (theta-update grad-batch theta-batch alpha lambda m-batch)]
                         (recur theta-batch (conj grad-vec (c/norm grad-batch)) (+ i 1)))
                       [theta-batch (mlu/avg grad-vec)]))]
               (recur new-theta grad (+ iter 1))))
         theta))))

  ; mini-batch processing with parallel gradient summation
  ([params data batch-size num-p]
   (let [[alpha lambda maxiter gtol] (unpack-params params)
         [X y] (unpack-data data)
         X (add-ones X)
         [m n] (c/size X)
         X-batches (mlu/get-mini-batches X batch-size)
         y-batches (mlu/get-mini-batches y batch-size)
         num-batches (count X-batches)]
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
                       (let [X-b (get X-batches i)
                             y-b (get y-batches i)
                             [m-batch n-batch] (c/size X)
                             X-p (mlu/get-batches X-b num-p)
                             y-p (mlu/get-batches y-b num-p)
                             data-p (map vector X-p y-p)
                             cost-prime-p (partial cost-prime theta-batch)
                             mapped (pmap cost-prime-p data-p)
                             grad-batch (reduce c/+ mapped)
                             theta-batch (theta-update grad-batch theta-batch alpha lambda m-batch)]
                         (recur theta-batch (conj grad-vec (c/norm grad-batch)) (+ i 1)))
                       [theta-batch (mlu/avg grad-vec)]))]
               (recur new-theta grad (+ iter 1))))
         theta))))
  )


; -----------------------
; Output

; return label predictions for data set X given parameters theta
(defn predict [X theta]
  (let [X (add-ones X)
        proba (compute-proba X theta)]
    (c/matrix (for [pred proba]
                (if (< pred 0.5) 0 1)))))

; return probability predictions for data set X given parameters theta
(defn predict-proba [X theta]
  (let [X (add-ones X)]
    (compute-proba X theta)))

; print each value in the probability matrix proba
(defn print-proba [proba]
  (let [[m n] (c/size proba)]
    (loop [i 0]
      (if (< i m)
        (do
          (println (mget proba i 0))
          (recur (+ i 1)))))))
