(ns index 
  (:require [scicloj.metamorph.ml.toydata :as toydata]
            [scicloj.metamorph.core :as mm]
            [libpython-clj2.python :as py]
            [tablecloth.api :as tc]
            [tech.v3.dataset.categorical :as ds-cat]
            [tech.v3.dataset.column-filters :as ds-cf]
            [scicloj.metamorph.ml :as ml]))



(require '[scicloj.metamorph.ml :as ml]
         '[scicloj.metamorph.core :as mm]
         '[scicloj.metamorph.ml.toydata :as toydata]
         '[tablecloth.api :as tc]
         '[tech.v3.dataset :as ds]
         '[tech.v3.dataset.categorical :as ds-cat]
         '[tech.v3.dataset.column-filters :as ds-cf]
         '[tech.v3.dataset.modelling :as ds-mod]
         '[libpython-clj2.python :refer [py. py.-] :as py])

;; # sklearn-clj
;; [sklearn-clj](https://github.com/scicloj/sklearn-clj) is a Clojure library
;; which allows to use all sklearn estimators (models and others) from Clojure.
;; It uses [libpython-clj](https://github.com/clj-python/libpython-clj) behind the scenes
;; but we do not need to use the `libpython-clj` API. All models are available via
;; the standard Clojure functions in `metamorph.ml`.


;; # Train sklearn model with `sklearn-clj`
;; In this scenario, we will not use any sklearn or libpython-clj API, only `metamorph.ml` functions

;; ## Use iris data
;; Lets first get our data, the well known iris dataset:
(def iris
  (-> (toydata/iris-ds)
      (ds-mod/set-inference-target :species)
      (ds/categorical->number [:species])))

;; ## Register models
;; This `require` will register all sklearn models and make 
;;them available to `metamorph.ml`
(require '[scicloj.sklearn-clj.ml]) 


;; ## Define metamorph pipeline
;; All models are available by specifying keys in form of :sklearn.xxx.yyy for the model type.
;; The available models are listed in the annex. They take the same parameters as in sklearn, just in kebap case.

;;
;; We define a normal `metamorph.ml` pipeline, as we would do with Clojure models.
(def pipe-fn
  (mm/pipeline
   {:metamorph/id :model}
   (ml/model {:model-type :sklearn.classification/logistic-regression
              :max-iter 1000
              :verbose true})))

;; It will use sklearn model "sklearn.linear_model.LogisticRegression"

;; ## Use tech.dataset as training data 
;; We need to train the model using a tech.ml.dataset as training data.
;; `sklearn-clj` will transform the data behind the scenes to a tech.v3.tensor, which libpython-clj auto-transforms to a numpy array , which the model can work with.
(def trained-ctx (mm/fit-pipe iris pipe-fn))
trained-ctx

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
;; `sklearn-clj` auto-transform the prediction result back to a
;; tech.ml.dataset
prediction

;; # Train model with sklearn using libpython-clj directly
;; As alternative approach we can use `libpython-clj` as well directly.
;; 
;; I take the following example an translate 1:1 into Clojure using libpython-clj
;; https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
;;
;; Import python modules as Clojure vars
(py/from-import sklearn.svm  SVC)
(py/from-import sklearn.preprocessing StandardScaler)
(py/from-import sklearn.datasets make_classification)
(py/from-import sklearn.model_selection train_test_split)
(py/from-import sklearn.pipeline Pipeline)
(py/import-as numpy np)


;; Define X and y from artifical data
(py/def-unpack [X y] ( make_classification  :random_state 0))

;; Split data in test and train
(py/def-unpack [X_train X_test y_train y_test] 
               ( train_test_split  X y :random_state 0))

;; we get 4 vars , example:
X_train


;; We define a pipeline with a standard scaler and a support vector machine model:
(def pipe ( Pipeline  [[ "scaler" ( StandardScaler)]  
                      [ "svc" ( SVC)] ]))
pipe

;; Train and score the pipeline:
(py/py.. pipe
         (fit  X_train y_train)
         (score  X_test, y_test))

;; Train and score the pipeline and set parameter:
(py/py.. pipe
         (set_params :svc__C 10)
         (fit  X_train y_train)
         (score  X_test, y_test))


;; # Train model with sklearn using libpython-clj from tech dataset
;; When we start with a tech.dataset, like

(-> iris tc/shuffle tc/head)

;; we need to first split it in :train and :test and convert it to row vectors
;; in (java) array format. Then libpython-clj knows how to convert these into
;; python (numpy)


;; Split in test and train:
(def train-test-split (tc/split->seq iris))

;; where data looks like this:
(-> train-test-split first :train)

;; a helper to call numpy.ravel() easier
(defn ravel [x]
  (py. np ravel x))

;; fit the pipeline to tech dataset, :train subset

(def fitted-pipe
  (py/py. pipe
          fit
          (-> train-test-split first :train ds-cf/feature tc/rows)
          (-> train-test-split first :train ds-cf/target tc/rows ravel)))


;; predict the pipeline to tech dataset :test subset
 (py/py. fitted-pipe predict
         (-> train-test-split first :test ds-cf/feature tc/rows))
         

;; score the model on :test data
(py/py. fitted-pipe score
        (-> train-test-split first :test ds-cf/feature tc/rows)
        (-> train-test-split first :test ds-cf/target tc/rows))
 
         
;; # Annex
;; List of model types of all sklearn models supported by sklearn-clj  (when using sklearn 1.5.1)
(->> @ml/model-definitions* keys sort) 
