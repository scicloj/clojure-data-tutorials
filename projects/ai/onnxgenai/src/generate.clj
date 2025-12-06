(ns generate
  (:import [ai.onnxruntime.genai SimpleGenAI]))

;; model is loaded by devcontaier setup
(def gen-ai (SimpleGenAI. "/tmp/models/microsoft/Phi-3-mini-4k-instruct-onnx/cuda/cuda-fp16/"))
(def gen-params (.createGeneratorParams gen-ai))
(.generate gen-ai 
           gen-params
           "What is onnxruntime-genai ?"
           ^java.util.function.Consumer (fn [s] (print s)))


