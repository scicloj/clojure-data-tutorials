
(load-string (slurp  "https://raw.githubusercontent.com/scicloj/clojure-data-tutorials/main/header.edn"))

^:kindly/hide-code
(ns improve-model
  (:require
   [clojure.data.csv :as csv]
   [clojure.java.io :as io]
   [clojure.string :as str]
   [scicloj.metamorph.ml.text :as text]
   [tablecloth.api :as tc]
   [tablecloth.column.api :as tcc]
   [scicloj.metamorph.ml :as ml]
   [scicloj.metamorph.ml.loss :as loss]
   [tech.v3.dataset.modelling :as ds-mod]
   ;[scicloj.clay.v2.api :as clay]
   [scicloj.ml.xgboost]
   [scicloj.metamorph.ml.classification]
   [scicloj.ml.smile.nlp :as nlp]
   [cheshire.core :as json]
   ))

(comment
  (require '[scicloj.clay.v2.api :as clay])
  (clay/start!)
  (clay/make! {:source-path "notebooks/improve-model.clj"
               :show false}))

(def stemmer (nlp/resolve-stemmer {}))

(defn stem [s]
  ( .stem stemmer s))

(defn tokenize-fn [text]
  (map (fn [token] (-> token str/lower-case stem))
       (str/split text #"\W+"))
  )



(defn- line-parse-fn [line]
  [(nth line 3)
   (Integer/parseInt (nth line 4))])

(def tidy-train
  (text/->tidy-text (csv/read-csv (io/reader "train.csv"))
                    seq
                    tokenize-fn
                    nlp/default-tokenize
                    :skip-lines 1))



(def tidy-train-ds
  (-> tidy-train :datasets first))

(-> tidy-train-ds :meta frequencies)
(def tfidf
  (->
   tidy-train-ds
   (text/->tfidf)
   (tc/rename-columns {:meta :label})))



(def for-split-calculations
  (tc/dataset {:document (-> tidy-train-ds :document distinct)}))

(def splits (tc/split->seq for-split-calculations))

(def train-ds
  (->
   (tc/left-join
    (-> splits first :train)
    tfidf
    :document)
   (ds-mod/set-inference-target [:label])))

(def test-ds
  (tc/left-join
   (-> splits first :test)
   tfidf
   :document))

(def n-sparse-columns (inc (tcc/reduce-max (train-ds :token-idx))))


(def model
  (ml/train train-ds {:model-type :xgboost/classification
                      :sparse-column :tfidf
                      :seed 123
                      :num-class 2
                      :n-sparse-columns n-sparse-columns}))
(def raw-prediction
  (ml/predict test-ds model))


(def test-true-labels
  (-> test-ds
      (tc/unique-by [:document :label])
      (tc/select-columns [:document :label])
      (tc/order-by :document)
      :label
      vec))

(def test-predicted-labels
  (mapv int
       (-> raw-prediction
           (tc/order-by :document)
           :label
           seq
           )))

(def acc
 (loss/classification-accuracy
  test-true-labels

  test-predicted-labels))

(spit "metrics.json"
      (json/encode {:acc acc}))



  