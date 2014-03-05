(ns logistic.core

	(:use 
		[compojure.core]
        [ring.middleware.session]
        [ring.middleware.session.cookie]
        [ring.middleware.content-type]
        [ring.middleware.params])

	(:require 
		[clojure.string :as str]
  		[clojure.tools.logging :as log]

        [ring.adapter.jetty :as jetty]
        [ring.middleware.resource :refer [wrap-resource]]
        [ring.util.response :as resp]
        [compojure.route :as route]
        [compojure.handler :as handler]

        [clatrix.core :as c]
        [clojure.core.matrix :as m]
        [clojure.core.matrix.operators :as mop]

        [logistic.mlutils :as mlu]
        [logistic.regression :as lr])

	(:gen-class :main true))

; ; parses a GET request query sting into a clojure Map object
; (import '[org.eclipse.jetty.util UrlEncoded MultiMap])
; (defn parse-query-string [query]
;   (let [params (MultiMap.)]
;     (UrlEncoded/decodeTo query params "UTF-8")
;     (into {} params)))


; define the routes for the app handler
(defroutes main-routes
	(GET "/" [] (resp/resource-response "logistic.html")))

(def app-handler 
	(-> main-routes
		(wrap-resource "public")
		compojure.handler/api))


; called upon server startup
(defn -main
	"Handles Client-Server Comunication"
	[& args]

	(println "Logistic -main started...")

	; load datasets
	(def train-data (mlu/load-data "data/spamdata/spamtrain.csv"))
	(def test-data (mlu/load-data "data/spamdata/spamtest.csv"))
	; (def train-data (mlu/load-data "data/trivialdata/train.csv"))
	; (def test-data (mlu/load-data "data/trivialdata/test.csv"))

	; scale data
	(def train-data (mlu/scale-features train-data -1.0 1.0))
	(def test-data (mlu/scale-features train-data -1.0 1.0))

	; init LR
	(def params (lr/get-params 0.8 0.0 1000))

	; fit model on training data
	(def theta (lr/fit params train-data 100 5))

	; predict labels and probabilities for testing data
	(def y-pred (lr/predict (test-data :data) theta))
	(def proba (lr/predict-proba (test-data :data) theta))

	; output result metrics
	(lr/print-proba proba)
	(def results (mlu/compute-accuracy (test-data :labels) y-pred))
	(println "Acurracy: " results)

	; output LR model

)


