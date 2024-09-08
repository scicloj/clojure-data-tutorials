(ns render
  (:require [scicloj.clay.v2.api :as clay]))

(clay/make! {:format [:quarto :html]
             :show false
             :base-source-path "notebooks"
             :source-path ["index.clj"
                           
                           ]
             :base-target-path "docs"
             :book {:title "Use sklearn-clj "}
             :clean-up-target-dir true})
(System/exit 0)

