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
	(def test-data (mlu/scale-features test-data -1.0 1.0))

	; init LR
	(def params (lr/get-params 0.8 0.0 1000))

	; fit model on training data
	; the last two parameters for lr/fit adjust mini-batch size and level of 
	; parallelism respectively
	; nil for either parameter implies no data splitting
	; Note: The fit function is overloaded so that unecessary splitting is not processes
	; if batch parameters are missing.
	(def theta (lr/fit params train-data 10))
	; an example of a fit call with mini-batch processing and parallelism is as follows:
	; (def theta (lr/fit params train-data 1000 20))

	; predict labels and probabilities for testing data
	(def y-pred (lr/predict (test-data :data) theta))
	(def proba (lr/predict-proba (test-data :data) theta))

	; output result metrics
	(lr/print-proba proba)
	(def results (mlu/compute-accuracy (test-data :labels) y-pred))
	(println "Acurracy: " results)

	; output LR model

)


