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
   [tablecloth.api :as tc]
   [tablecloth.column.api :as tcc]
   [taoensso.nippy :as nippy]
   [tech.v3.dataset :as ds]
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
  (.stem stemmer s))

(defn tokenize-fn [text]
  (let [cleaned-text
        (str/replace text #"(http:|https:)+[^\s]+[\w]" "")]
    (->>
     (str/split cleaned-text #"\W+")
     (map (fn [token]
            (let [lower-case (-> token str/lower-case)
                  non-stop-word
                  (if (contains? stop-words lower-case)
                    ""
                    lower-case)]
              (-> non-stop-word stem))))
     (remove empty?))))

(defn- line-parse-fn [line]
  [(str
    (nth line 1)
    " "
    (nth line 2)
    " "
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



(def for-split-calculations
  (tc/dataset {:document (-> tidy-train-ds :document distinct)}))

(def splits (tc/split->seq for-split-calculations :holdout {:seed 123}))


(def train-ds
  (->
   (tc/left-join
    (-> splits first :train)
    tfidf
    :document)
   (ds-mod/set-inference-target [:label])
   (tc/select-columns [:document :tfidf :token-idx :label])
   (tc/order-by [:document :token-idx])
   (tc/clone)))



(def test-ds
  (->
   (tc/left-join
    (-> splits first :test)
    tfidf
    :document)
   (tc/clone)))

(def test-true-labels
  (-> test-ds
      (tc/unique-by [:document :label])
      (tc/select-columns [:document :label])
      (tc/order-by :document)
      :label
      vec))

(defn dissoc-in
  [m [k & knext :as ks]]
  (cond
    (and knext
         (contains?
          (get-in m (butlast ks))
          (last ks))) (update-in m (butlast ks) dissoc (last ks))
    (not knext) (dissoc m k)
    :else m))

(defn- score [opts train-ds test-ds test-true-labels]
  (let [
        n-sparse-columns (inc (tcc/reduce-max (train-ds :token-idx)))

        model-opts

        (merge opts
               {:model-type :xgboost/classification
                :sparse-column :tfidf
                :seed 123
                :num-class 2
                :verbosity 2
                :validate_parameters true
                :n-sparse-columns n-sparse-columns})
        
        model 
        (->
         (ml/train train-ds model-opts)
         (dissoc-in [:model-wrapper :id]))
        raw-prediction (ml/predict test-ds model)
        test-predicted-labels
        (mapv int
              (-> raw-prediction
                  (tc/order-by :document)
                  :label
                  seq))

        acc
        (loss/classification-accuracy
         test-true-labels
         test-predicted-labels)]
     (println :acc acc)
     {:opts model-opts
      :acc acc}))

(def results
  (->> (ml/hyperparameters :xgboost/classification)
       (grid/sobol-gridsearch)
       (take 5)
       (map #(score % train-ds test-ds test-true-labels))
       (doall)))





;; (spit "metrics.json"
;;       (json/encode {:acc acc}))


;; (let [tfidf-test-ds
;;       (->
;;        (text/->tidy-text (csv/read-csv (io/reader "test.csv"))
;;                          seq
;;                          (fn [line]
;;                            [(nth line 3) {:id (first line)}])
;;                          tokenize-fn
;;                          :skip-lines 1
;;                          :new-token-behaviour :as-unknown
;;                          :token->index-map train--token-lookup-table)
;;        :datasets
;;        first
;;        text/->tfidf
;;        (tc/select-columns [:document :token-idx :tfidf :meta])
;;    ;; the :id for Kaggle
;;        (tc/add-column
;;         :id (fn [df] (map
;;                       #(:id %)
;;                       (:meta df))))
;;        (tc/drop-columns [:meta]))]

;;   (->
;;    (ml/predict tfidf-test-ds model)
;;    (tc/right-join tfidf-test-ds :document)
;;    (tc/unique-by [:id :label])
;;    (tc/select-columns [:id :label])
;;    (tc/update-columns {:label (partial map int)})
;;    (tc/rename-columns {:label :target})
;;    (tc/write-csv! "submission.csv")))



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
       (take 50)
       (queue-exp)))




(comment




  (k/exists? store "hello" {:sync? true})

  (hash tidy-train)
  ;;=> 1006577141

  (hash train-ds)
  ;;=> 866734393

  (hash [1.0 1.1])
  ;;=> -430348069

  (hash [1.123456 1.1234567])

  (def x
    (time
     (ml-cache/caching-train store train-ds opts)))

  (def x
    (time
     (def model
       (ml/train train-ds opts))))

  (time
   (let [bytes (nippy/freeze model)
         model-2 (nippy/thaw bytes)]
     {:hash
      (hash
       {:train-ds train-ds
        :options opts})
      :bytes bytes
      :model model-2})))
