(ns index 
  (:require [scicloj.metamorph.ml.toydata :as toydata]
            [scicloj.metamorph.core :as mm]
            [libpython-clj2.python :as py]))


(require '[scicloj.sklearn-clj.ml]) ;; registers all models
(require '[scicloj.metamorph.ml :as ml]
         '[scicloj.metamorph.core :as mm]
         '[scicloj.metamorph.ml.toydata :as toydata]
         '[tablecloth.api :as tc]
         '[tech.v3.dataset :as ds]
         '[tech.v3.dataset.categorical :as ds-cat]
         '[tech.v3.dataset.modelling :as ds-mod]
         '[libpython-clj2.python :refer [py. py.-]])

;; # Introduction

;; # Use iris data
(def iris
  

  (-> (toydata/iris-ds)
      (ds-mod/set-inference-target :species)
      (ds/categorical->number [:species])))
  

;; # Train sklearn model
;; ## Define metamorph pipeline
(def pipe-fn
  (mm/pipeline
   {:metamorph/id :model}
   (ml/model {:model-type :sklearn.classification/logistic-regression
              :max-iter 1000})))

;; ## Train model
(def trained-ctx (mm/fit-pipe iris pipe-fn))


;; ## Inspect model trained model
(def model-object
  (-> trained-ctx :model :model-data :model))
;; Its a libpython-clj reference to a python object
model-object

;; We can get its coeffiecints, for example
(class
 (py/->jvm
  (py.- model-object coef_)))
         


