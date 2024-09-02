(ns render
  (:require [scicloj.clay.v2.api :as clay]))

(clay/make! {:format [:quarto :html]
             :show false
             :base-source-path "notebooks"
             :source-path ["index.clj"
                           "onnx.clj"
                           ]
             :base-target-path "docs"
             :book {:title "Using ONNX models from clojure"}
             :clean-up-target-dir true})
(System/exit 0)
