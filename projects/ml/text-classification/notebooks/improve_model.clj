(load-string (slurp  "https://raw.githubusercontent.com/scicloj/clojure-data-tutorials/main/header.edn"))

^:kindly/hide-code
(ns improve-model
  (:require
   [cheshire.core :as json]
   [clojure.data.csv :as csv]
   [clojure.java.io :as io]
   [clojure.java.shell :as sh]
   [clojure.pprint :as pprint]
   [clojure.string :as str]
   [scicloj.metamorph.ml :as ml]
   [scicloj.metamorph.ml.classification]
   [scicloj.metamorph.ml.loss :as loss]
   [scicloj.metamorph.ml.text :as text]
   [scicloj.metamorph.ml.gridsearch :as grid]
   [scicloj.ml.smile.nlp :as nlp]
   [scicloj.ml.xgboost]
   [ham-fisted.reduce :as hf-reduce]

[tech.v3.dataset.reductions :as ds-reduce]   
   [tablecloth.api :as tc]
   [tablecloth.column.api :as tcc]
   [tech.v3.dataset.modelling :as ds-mod] ;[scicloj.clay.v2.api :as clay]
   
   [scicloj.metamorph.ml.gridsearch :as ml-gs]))




(comment
  (require '[scicloj.clay.v2.api :as clay])
  (clay/start!)
  (clay/make! {:source-path "notebooks/improve-model.clj"
               :show false}))

(def stop-words
  (into #{}
        (iterator-seq
         (.iterator
          smile.nlp.dictionary.EnglishStopWords/COMPREHENSIVE))))

(def stemmer (nlp/resolve-stemmer {}))

(defn stem [s]
  ( .stem stemmer s))

(defn tokenize-fn [text]
  (map (fn [token] 
         (let [lower-case (-> token str/lower-case)
               non-stop-word 
               (if ( contains? stop-words lower-case)
                 ""
                 lower-case
                 )
               ]
           
           (-> non-stop-word stem)))
       (str/split text #"\W+"))
  )


(defn- line-parse-fn [line]
  [(str 
    (nth line 1)
    " - "
    (nth line 2)
    " - "
    (nth line 3))
   (Integer/parseInt (nth line 4))])

(def tidy-train
  (text/->tidy-text (csv/read-csv (io/reader "train.csv"))
                    seq
                    line-parse-fn
                    tokenize-fn
                    :skip-lines 1))



(def tidy-train-ds
  (-> tidy-train :datasets first))


(def tfidf
  (->
   tidy-train-ds
   (text/->tfidf)
   (tc/rename-columns {:meta :label})))

;(-> tfidf :document distinct count)
;;=> 7613

;; (def more-then-x
;;   (->
;;    (ds-reduce/group-by-column-agg 
;;     :token-idx
;;     {:c (ds-reduce/row-count)}
;;     tfidf
;;     )
;;    (tc/select-rows (fn [row] (or (< (:c row) 5000)
;;                                  (> (:c row) 20))))
;;    (tc/order-by :c)
;;    (tc/unique-by :token-idx)
;;    ))

;; (def tfidf
;;   (-> more-then-x
;;       (tc/left-join tfidf :token-idx)
;;       (tc/drop-missing)))

(def for-split-calculations
  (tc/dataset {:document (-> tfidf :document distinct)}))

(def splits (tc/split->seq for-split-calculations :holdout {:seed 123}))

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
  (ml/train train-ds 
            (merge (:predict (json/parse-stream (io/reader "params.json") keyword))
                   {:model-type :xgboost/classification
                    :sparse-column :tfidf
                    :seed 123
                    :num-class 2
                    :validate_parameters true
                    :n-sparse-columns n-sparse-columns})
            ))
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

(println :acc acc)
(spit "metrics.json"
      (json/encode {:acc acc}))

(comment
  (defn queue-exp
    "Queue a number of experiments with dvc.
    The provided `params` (list of maps) is converted to yaml and
    written to `params.yaml` and a new job is queued using it.
  
    The list of params can for example be created from [[scicloj.metamorph.ml.gridsearch/sobol-gridsearch]]
    "
    [params]
    (run!
     #(do (spit "params.json" (json/generate-string %))
          (pprint/pprint
           (sh/sh "dvc" "exp" "run" "--queue")))
     params))

  (->> (ml/hyperparameters :xgboost/classification)
       (grid/sobol-gridsearch)
       (take 10)
       (queue-exp)
       )
  
  )

  