;;(load-file "../../header.edn")

^:kindly/hide-code
(ns index)

;; # Preface

;; [ONNX](https://onnxruntime.ai/) is an upcoming exchange format for machine learning models.
;; It is platform indepedent and allows, for example, to train models in python, export the trained model
;; into a file, and then use it in Java / Clojure for inference