(ns logistic.mlutils
  (:require
    [clojure.math.numeric-tower :as math]
    [clojure.data.csv :as csv]
    [clojure.java.io :as io]
    [clatrix.core :as c]))

(defn strings-to-doubles
  "convert all strings in a matrix to Doubles"
  [c]
  (for [l c]
    (for [v l]
      (Double/valueOf v))))

(defn submat-cols
  "returns a submatrix given a range of column indices start s and end e"
  [data s e n]
  (if (and (>= 0 s) (< s e) (<= e n))
    (loop [res (c/slice data _ s)
           j (+ 1 s)]
      (if (< j e)
        (recur (c/hstack res (c/slice data _ j)) (+ 1 j))
        res))
    nil))

(defn randomize
  "randomizes dataset instances"
  [data]
  (let [[m n] (c/size data)]
    (if (> m 0)
      (let [rand-indices (vec (shuffle (range m)))]
        (loop [res (c/slice data (get rand-indices 0) _)
               i 1]
          (if (< i m)
            (recur (c/vstack res (c/slice data (get rand-indices i) _)) (+ i 1))
            res))))))

(defn split-dataset
  "split dataset into separate data/labels matrices"
  [data]
  (let [[m n] (c/size data)]
    {:data   (submat-cols data 0 (- n 1) n)
     :labels (c/slice data _ (- n 1))}))

(defn load-data
  "loads a csv file into nested lists of Doubles"
  [in-file]
  (->> in-file
       (io/reader)
       (csv/read-csv)
       (strings-to-doubles)
       (c/matrix)
       (randomize)
       (split-dataset)))

; TODO: scales each feature in the data set to values between given min and max
; python code:
;	X_min, X_max = X.min(0), X.max(0)
;	return (((X - X_min) / (X_max - X_min)) * (new_max - new_min + 0.000001)) + new_min
; currently using clatrix normalization method
(defn scale-features [{:keys [data labels]} new-min new-max]
  {:data (c/normalize data)
   :labels labels})

(defn compute-accuracy
  "returns the percentage of correct predictions"
  [y y-pred]
  (let [[m n] (c/size y)
        diff (c/- y y-pred)
        correct (count (filter #{0.0} diff))]
    (float (/ correct m))))

(defn get-batch
  "slices rows from matrix mat with starting index s and end e"
  [mat s e]
  (loop [batch (c/slice mat s _)
         i (+ s 1)]
    (if (< i e)
      (recur (c/vstack batch (c/slice mat i _)) (+ i 1))
      batch)))

(defn get-batches
  "return a vector of b batches sliced from matrix mat"
  [mat b]
  (if (= b nil) [mat]
                (let [[m n] (c/size mat)
                      div (math/ceil (float (/ m b)))]
                  (if (> b m) [mat]
                              (loop [i 0 j b res []]
                                (cond
                                  (and (< i m) (< j m))
                                  (recur j (+ j div) (conj res (get-batch mat i j)))
                                  (and (< i m) (>= j m))
                                  (recur j (+ j div) (conj res (get-batch mat i m)))
                                  :else
                                  res))))))

(defn get-mini-batches
  "returns a vector of b size batches sliced from matrix mat"
  [mat b]
  (if (= b nil) [mat]
                (let [[m n] (c/size mat)]
                  (if (>= b m) [mat]
                               (loop [i 0 j b res []]
                                 (cond
                                   (and (< i m) (< j m))
                                   (recur j (+ j b) (conj res (get-batch mat i j)))
                                   (and (< i m) (>= j m))
                                   (recur j (+ j b) (conj res (get-batch mat i m)))
                                   :else
                                   res))))))

(defn avg
  "returns the mean of vector v"
  [v]
  (let [c (count v)]
    (float (/ (reduce + v) c))))


