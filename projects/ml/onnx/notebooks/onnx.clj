(ns onnx
  (:require [tablecloth.api :as tc]
            [clojure.java.data :as j])
  (:import [ai.onnxruntime OrtEnvironment 
            OnnxTensor]))

;; # Using ONNX models from Clojure
;;
;; We assume that we get an ONNX file from somewhere and want to use it from Clojure.
;; A model can for example be trained with Python sklearn and exported to ONNX format.
;; There are as well model zoos existing, which allow to download pre-trained models on ONNX format.
;;
;; We can then open such file using the JAVA ONNX run-time which has the maven coordinates
;; com.microsoft.onnxruntime/onnxruntime {:mvn/version "1.19.0"}

;; ## Load and inspect ONNX file
;; We use here a model which was trained on the well known iris data and can predict the species
(def env (OrtEnvironment/getEnvironment))
(def session (.createSession env "logreg_iris.onnx"))



;;  We can inspect the model and among other information discover which input format it needs.


(j/from-java-deep
 (.getInputInfo session)
 {})

;; This shows us that it has one input called "float_input" which needs to be a 2D tensor
;; with dimensions (anyNumber, 4)
;; This matches our knowledge on the iris data, which has 4 columns (+ prediction)
;;
;; In a similar way we can introspect the model output returned on inference:

(j/from-java-deep
 (.getOutputInfo session)
 {})

;; This model predicts one value for each row of the input, which matches as well the iris data.

;; Now we need to construct an instance of ai.onnxruntime.OnnxTensor of shape [anyNumber,4]
;; This can be done starting from a vector-of-vector, for example

;; ## Run inference on arrays
(def input-data
  [[7   0.5  0.5 0.5]
   [0.5 1    1     1]])

(def tensor (OnnxTensor/createTensor
             env
             (into-array (map float-array input-data))))

tensor

(def prediction (.run session {"float_input" tensor}))
prediction

;; We have two pieces of data in the prediction result:
(map key prediction)

;; namely predicted labels and probabilities

;; We need a bit of inter-op to get the numbers out of the prediction


;; predicted species:
(->  prediction first val .getValue)
;; probability distribution for each species for all labels:
(map
 #(.getValue %)
 (->  prediction second val .getValue))

;; ## Run inference on tech.dataset
;; In case we have our data in a tech.ml.dataset

(def ds
  (tc/dataset [[0.5 0.5 0.5 0.5]
               [1   1   1   1]
               [1   1   2   7]
               [3   1   2   1]
               [7   8   2   10]]))

ds
;; we can convert it to a OnnxTensor as well easily

(def tensor-2 
  (OnnxTensor/createTensor 
      env 
     (into-array (map float-array (tc/rows ds)))))

;; Running predictions is then the same.
(def prediction-2 (.run session {"float_input" tensor-2}))
(.. prediction-2 (get 0) getValue)

;; Overall we can use any ONNX model from Clojure.
;; This allows polyglot scenarios where data preprocession and model evaluation
;; is done in Clojure, while training is done in Python with its huge ecosystem of ML models.
;;
;; Hopefuly over time the ONNX standard will see widespread use. Most sklearn models/pipelines can be exported 
;; to ONNX using [sklearn-onnx](https://onnx.ai/sklearn-onnx/)
;; Other python ML frmeworks start to support ONNX as well for example PyTorch,
;; see [PyTorch ONNX export](https://pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html)
