(ns onnx
  (:require [tablecloth.api :as tc]
            [clojure.java.data :as j])
  (:import [ai.onnxruntime OrtEnvironment OrtSession$SessionOptions
            OnnxTensor]))

;; We assume that we get an ONNX file from somewhere and wnat to use it from Clojure.
;; ONXX files get for example be trained with Python sklearn and exported to ONNX format
;; There are as well model zoos existing, which allow to download pretrained models
;;
;; We can open such file using the JAVA ONNX runtime  with maven coordinates 
;; com.microsoft.onnxruntime/onnxruntime {:mvn/version "1.19.0"}

;; We use here a model which was trained on te well know iris, data and can predict the species
(def env (OrtEnvironment/getEnvironment))
(def session (.createSession env "logreg_iris.onnx"))



;;  We can inspect the model and among ither things discoer which input format it needs.


(j/from-java-deep
 (.getInputInfo session)
 {})

;; This shows us that it has one input called "float_input" which needs to be a 2D tensor
;; with dimensions (anyNumber, 4)
;; This matches our knowledge on the iris data, which has 4 columns (+ prediction)
;;
;; In a similar way we can introspect the output after inference:

(j/from-java-deep
 (.getOutputInfo session)
 {})

;; It will output one value for each row of the input, which matches as well the iris data.

;; Now we need to construct an instance of ai.onnxruntime.OnnxTensor of shape [-1,4]
;; This can be done starting from a vector-of-vector, for example

(def input-data
  [[7   0.5  0.5 0.5]
   [0.5 1    1     1]])

(def tensor (OnnxTensor/createTensor
             env
             (into-array (map float-array input-data))))

tensor

(def prediction (.run session {"float_input" tensor}))
prediction

;; we have two things in prediction
(map key prediction)

;; predicted lables and probabilities

;; We need a bit of interop to get the numbers out of the prediction 


;; predicted species
(->  prediction first val .getValue)
;; probablity distribution for each species for all labels
(map
 #(.getValue %)
 (->  prediction second val .getValue))

;; In case we ave our data to be used in a tech.ml.dataset

(def ds
  (tc/dataset [[0.5 0.5 0.5 0.5]
               [1 1 1 1]
               [1 1 2 7]
               [3 1 2 1]
               [7 8 2 10]]))


;; we can convert it to a tensor as well easely

(def tensor-2 
  (OnnxTensor/createTensor 
      env 
     (into-array (map float-array (tc/rows ds)))))

(def prediction-2 (.run session {"float_input" tensor-2}))
(.. prediction-2 (get 0) getValue)

