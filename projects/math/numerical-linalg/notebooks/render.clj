(ns render
  (:require [scicloj.clay.v2.api :as clay]))

(clay/make! {:format [:quarto :html]
             :show false
             :base-source-path "notebooks"
             :source-path ["index.clj"
                           "svd.clj"]
             :base-target-path "docs"
             :book {:title "Numerical Linear Algebra"}
             :clean-up-target-dir true})
(System/exit 0)
