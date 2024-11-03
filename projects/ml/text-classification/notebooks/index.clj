(load-string (slurp  "https://raw.githubusercontent.com/scicloj/clojure-data-tutorials/main/header.edn"))

^:kindly/hide-code
(ns index 
  (:require
   [clojure.data.csv :as csv]
   [clojure.java.io :as io]
   [clojure.string :as str]
   [scicloj.metamorph.ml.text :as text]
   [tablecloth.api :as tc]
   [tablecloth.column.api :as tcc]
   [scicloj.metamorph.ml :as ml]
   [tech.v3.dataset.modelling :as ds-mod]
   [scicloj.clay.v2.api :as clay]))

(comment
  (require '[scicloj.clay.v2.api :as clay])
  (clay/start!)
  (clay/make! {:source-path "notebooks/index.clj"
               :show false
               }))

;; The following code shows how to perform text classification from a Kaggle 
;; dataset and make a submission file, ready to get uploaded to 
;;Kaggle for scoring.
;;
;; It makes use of the tidy text / TFIDF functionality present in `metamorph.ml`
;; and the ability of the xgboost model to handle tidy text data as input.
;;
;; First we need a fn to tokenize a line of text
;; The simplest such function is:
(defn- tokenize-fn [text]
  (str/split text #" "))


;; It does not do any text normalization, which is always required in NLP tasks
;; in order to have a more general model.
;;
;; The following reads line-by-line a file from disk and converts it on the fly
;; to the  `tidy text` representation, it which each word 
;; is a row in a dataset.
;;
;; `line-parse-fn` needs to split an input line into [text meta],
;; and the `text` is then further handled by `tokenize-fn` and split into tokens.
;; The format of the data has the text in field 4 and the label in 5.

;; We ignore all other columns so far:
(defn- line-parse-fn [line]
  [(nth line 3)
   (Integer/parseInt (nth line 4))])

;; This triggers the parsing and produces a (seq of) "long" datasets
;; (1 for our small text)
;; and the vocabulary obtained during parsing.
(def tidy-train
  (text/->tidy-text (csv/read-csv (io/reader "train.csv"))
                    seq
                    line-parse-fn
                    tokenize-fn
                    :skip-lines 1))

(def tidy-train-ds 
  (-> tidy-train :datasets first))
;; The combination of columns :document, :token-pos and :token-index
;; together with the vocabulary table is an exact representation of the text
;; Unless we normalize it as part of hte `tokenize-fn`
;;
;; `meta` is any other information of a row to be kept, usualy the "label"
;; in case of training data.

tidy-train-ds

;; The lookup table allow to convert from :token-idx to words and 
;;back if needed.
(def train--token-lookup-table (:token-lookup-table tidy-train))
(map str (take 20 train--token-lookup-table))

;; As we can see, the tokens are not cleaned / standardized at all. 
;;This gives as well a large vocabulary size of
(count train--token-lookup-table)


;; Now we convert the text into a bag-of-words format, which looses
;; any word order and calculates a metric which is known to work well
;; for text classification, the so called TFIDF score.
(def train-tfidf
  (text/->tfidf tidy-train-ds))

;; The resulting table represent conceptually well three "sparse matrices"
;; where :document and :token-idx are  x,y coordinates and matrix cell values
;; are :token-count, term-frequency (:tf) or TFIDF
;;
;; Not present rows (the large majority) are 0 values.

;; A subset of machine learning algorithms can deal with sparse matrices, 
;; without then need to convert them into
;; dense matrices first, which is in most cases impossible due to the memory
;; consumption

;; The train-tfidf dataset represents therefore  3 sparse matrices with
;; dimensions
(tcc/reduce-max (:document train-tfidf))
;; times
(tcc/reduce-max (:token-idx train-tfidf))
;; time 3
;; =

(* (tcc/reduce-max (:document train-tfidf))
   (tcc/reduce-max (:token-idx train-tfidf))
   3)


;; while only having shape:

(tc/shape train-tfidf)

;; This is because most matrix elements are 0, as 
;; any text does "not contain" most words.
;;
;; As TFIDF (and its variants) are one of the most common numeric representations for text,
;; "sparse matrixes" and models supporting them is a prerequisite for NLP.
;;
;; Only since a few years we have "dense text representations" based on "embeddings",
;; which will not be discussed here today,

;; Now we get the data ready for training.

(def train-ds
  (-> train-tfidf
      (tc/rename-columns {:meta :label})
      (tc/select-columns [:document :token-idx :tfidf :label]) ;; we only need those
      (ds-mod/set-inference-target [:label])))

train-ds

(def n-sparse-columns (inc (tcc/reduce-max (train-ds :token-idx))))

;; The model used is from library `scicloj.ml.xgboost` which is the well known xgboost model
;; behind a wrapper to make it work with tidy text data.
;;
;; We use :tfidf column as the "feature".

(require '[scicloj.ml.xgboost]) 
;; registers the mode under key :xgboost/classification

(def model
  (ml/train train-ds {:model-type :xgboost/classification
                         :sparse-column :tfidf
                         :seed 123
                         :num-class 2
                         :n-sparse-columns n-sparse-columns}))


;; Now we have a trained model, which we can use for prediction on the test data.

;; This time we do parsing and tfidf in one go.
;;
;; Important here:
;;
;; We pass the vocabulary "obtained before" in order to be sure, that
;; :token-idx maps to the same words in both datasets. In case of "new tokens",
;; we ignore them and map them to a special token, "[UNKNOWN]"
(def tfidf-test-ds
  (->
   (text/->tidy-text (csv/read-csv (io/reader "test.csv"))
                     seq
                     (fn [line]
                       [(nth line 3) {:id (first line)}])
                     tokenize-fn
                     :skip-lines 1
                     :new-token-behaviour :as-unknown
                     :token->index-map train--token-lookup-table)
   :datasets
   first
   text/->tfidf
   (tc/select-columns [:document :token-idx :tfidf :meta]) 
   ;; he :id for Kaggle
   (tc/add-column
    :id (fn [df] (map
                  #(:id %)
                  (:meta df))))
   (tc/drop-columns [:meta])))

;; This gives the dataset which can be passed into the `predict` function of `metamorph.ml`
tfidf-test-ds

(def prediction
  (ml/predict tfidf-test-ds model))

;; The raw predictions contain the "document" each prediction is about.
;; This we can use to match predictions and the input "ids" in order to produce teh format
;; required by Kaggle
prediction

(->
 (tc/right-join prediction tfidf-test-ds :document)
 (tc/unique-by [:id :label])
 (tc/select-columns [:id :label])
 (tc/update-columns {:label (partial map int)})
 (tc/rename-columns {:label :target})
 (tc/write-csv! "submission.csv"))

;; The produced CVS file can be uploaded to Kaggle for scoring.

(->>
 (io/reader "submission.csv")
 line-seq
 (take 10))