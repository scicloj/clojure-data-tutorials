(ns generate
  (:import [ai.onnxruntime.genai SimpleGenAI]))

;; model is loaded by devcontaier setup
;;onnx-community/gpt-oss-20b-ONNX
(with-open [gen-ai 
            (SimpleGenAI. "/tmp/models/microsoft/Phi-3-mini-4k-instruct-onnx/cuda/cuda-fp16/")
            ;(SimpleGenAI. "/tmp/models/onnx-community/gpt-oss-20b-ONNX")
            
            gen-params (.createGeneratorParams gen-ai)]
  (.generate gen-ai 
             gen-params
             "Invent a hyphothetical season of Italian Serie A matches.
              Return in JSON format all matches of Serie A, with realistic outcomes.
              Each match should be in JSON, containing:
              1. both teams
              2. match result
              3. goals
              4. scorer and minutes of goals
              "
             ^java.util.function.Consumer (fn [s] (print s))))


