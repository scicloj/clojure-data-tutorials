(ns sklearn 
  (:require [scicloj.metamorph.ml.toydata :as toydata]
            [scicloj.metamorph.core :as mm]))


(require '[scicloj.sklearn-clj.ml]) ;; registers all models
(require '[scicloj.metamorph.ml :as ml]
         '[scicloj.metamorph.core :as mm]
         '[scicloj.metamorph.ml.toydata :as toydata]
         '[tablecloth.api :as tc]
         '[tech.v3.dataset :as ds]
         '[tech.v3.dataset.categorical :as ds-cat]
         '[tech.v3.dataset.modelling :as ds-mod]
         '[libpython-clj2.python :refer [py. py.-]])

(def iris
  

  (-> (toydata/iris-ds)
      (ds-mod/set-inference-target :species)
      (ds/categorical->number [:species])))
  

(def pipe-fn
  (mm/pipeline
   {:metamorph/id :model}
   (ml/model {:model-type :sklearn.classification/logistic-regression
              :max-iter 1000})))

(def trained-ctx (mm/fit-pipe iris pipe-fn))

(def model-object
  (-> trained-ctx :model :model-data :model))

(println :model-object model-object)
(println :coeff
         (py.- model-object coef_))


