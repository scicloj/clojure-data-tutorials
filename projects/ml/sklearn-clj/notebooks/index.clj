(ns index 
  (:require [scicloj.metamorph.ml.toydata :as toydata]
            [scicloj.metamorph.core :as mm]
            [libpython-clj2.python :as py]
            [tablecloth.api :as tc]
            [tech.v3.dataset.categorical :as ds-cat]))



(require '[scicloj.metamorph.ml :as ml]
         '[scicloj.metamorph.core :as mm]
         '[scicloj.metamorph.ml.toydata :as toydata]
         '[tablecloth.api :as tc]
         '[tech.v3.dataset :as ds]
         '[tech.v3.dataset.categorical :as ds-cat]
         '[tech.v3.dataset.modelling :as ds-mod]
         '[libpython-clj2.python :refer [py. py.-]])

;; # Introduction
;; [sklearn-clj](https://github.com/scicloj/sklearn-clj) is a Clojure librray
;; which allows to use all sklearn estimators (models and others) from Clojure
;; It use [libpython-clj](https://github.com/clj-python/libpython-clj) behind the scenes
;; but we do not need to use the `libpython-clj` API. All models are available via
;; the standart Clojure functions in `metamorph.ml`

;; # Use iris data
;; Lets first get our data, the well kownn iris dataset
(def iris
  

  (-> (toydata/iris-ds)
      (ds-mod/set-inference-target :species)
      (ds/categorical->number [:species])))

;; # Register models
;; This `require` will register all sklearn models and make 
;;them available to `metamorph.ml`
(require '[scicloj.sklearn-clj.ml]) 

;; # Train sklearn model
;; ## Define metamorph pipeline
;; All models are available by specifying keys in form of :sklearn.xxx.yyy for the model type.
;; The available models are listed [here](https://scicloj.github.io/scicloj.ml-tutorials/userguide-sklearnclj.html)
;; (The list is a bit old, from sklearn 1.0.0)
;;
;; We define a normal `metamorph.ml` pipeline, as we would to with Clojure models.
(def pipe-fn
  (mm/pipeline
   {:metamorph/id :model}
   (ml/model {:model-type :sklearn.classification/logistic-regression
              :max-iter 1000})))

;; ## Train model
;; We can train the model using a tech.ml.dataset as training data.
;; `sklearn-clj` will transform the data behind the scenes to numpy arrays, which
;; the sklearn models expect.
(def trained-ctx (mm/fit-pipe iris pipe-fn))


;; ## Inspect trained model
;; We can inspect the model object:
(def model-object
  (-> trained-ctx :model :model-data :model))
;; It's a libpython-clj reference to a python object
model-object

;; and use `libpython-clj` functions to get information out of it.
;; We can get the models coefficients, for example:
(py/->jvm
  (py.- model-object coef_))

;; we can as well ask for predict on new data
(def simulated-new-data 
  (tc/head (tc/shuffle iris) 10) )

(def prediction
  (->
   (:metamorph/data
    (mm/transform-pipe 
     simulated-new-data  
     pipe-fn 
     trained-ctx))
   ds-cat/reverse-map-categorical-xforms))

;; We get a tech.ml.dataset with the prediction result back.
;; `sklearn-clj` aut-transform the prediction result back to a
;; tech.ml.dataset
prediction


