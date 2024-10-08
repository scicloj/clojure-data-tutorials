
;; This tutorial demonstrates using
;; the probabilistic programming library PyMC
;; from Clojure.

;; See the [Introductory Overview of PyMC](https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/pymc_overview.html).

;; ## Setup

;; Relevant Clojure namespaces:

(ns index
  (:require [libpython-clj2.require :refer [require-python]]
            [libpython-clj2.python :refer [py. py.. py.-] :as py]
            [fastmath.random :as random]
            [tablecloth.api :as tc]
            [tablecloth.column.api :as tcc]
            [tech.v3.datatype :as dtype]
            [scicloj.hanamicloth.v1.plotlycloth :as ploclo]
            [scicloj.kind-pyplot.v1.api :as pyplot]
            [scicloj.kindly.v4.kind :as kind]))

;; Relevant Python modules:

(require-python '[builtins :as python]
                'operator
                '[arviz :as az]
                '[arviz.style :as az.style]
                '[pandas :as pd]
                '[matplotlib.pyplot :as plt]
                '[numpy :as np]
                '[numpy.random :as np.random]
                '[pymc :as pm])

;; Some convenience functions to access Python idioms: 

(defn brackets [obj entry]
  (py. obj __getitem__ entry))

(def colon
  (python/slice nil nil))

;; Theme for ArViZ visualizations:

;; ## Synthetic data

(arviz.style/use "arviz-darkgrid")

(def random-seed 8927)

(def dataset-size 101)

(def true-parameter-values
  {:alpha 1
   :sigma 1
   :beta [1 2.5]})

(defn gen-dataset [{:keys [size random-seed
                           alpha sigma beta]}]
  (let [rng (random/rng :isaac random-seed)]
    (-> {:x1 (take size (random/->seq rng))
         :x2 (-> (take size (random/->seq rng))
                 (tcc/* 0.2))}
        tc/dataset
        (tc/add-column :y
                       #(-> (tcc/+ alpha
                                   (tcc/* (beta 0) (:x1 %))
                                   (tcc/* (beta 1) (:x2 %))
                                   (tcc/* sigma
                                          (dtype/make-reader
                                           :float32 size (rand)))))))))



(def dataset
  (gen-dataset (merge {:random-seed random-seed
                       :size dataset-size}
                      true-parameter-values)))

(->> [:x1 :x2]
     (mapv (fn [x]
             (-> dataset
                 (ploclo/layer-point
                  {:=x :x1}))))
     kind/fragment)

pm/__version__


(def basic-model (pm/Model))

(py/with [_ basic-model]
         (let [{:keys [x1 x2 y]} (-> dataset
                                     (update-vals np/array))
               alpha (pm/Normal "alpha"
                                :mu 0
                                :sigma 10)
               beta (pm/Normal "beta"
                               :mu 0
                               :sigma 10
                               :shape 2)
               sigma (pm/HalfNormal "sigma"
                                    :sigma 1)
               mu (operator/add alpha
                                (operator/mul (brackets beta 0)
                                              x1)
                                (operator/mul (brackets beta 0)
                                              x2))
               y_obs (pm/Normal "y_obs"
                                :mu mu
                                :sigma sigma
                                :observed y)]))

(def idata
  (py/with [_ basic-model]
           (pm/sample)))


(-> idata
    (py.- posterior)
    (py.- alpha)
    (py. sel :draw (python/slice 0 4)))


(def slice-idata
  (py/with [_ basic-model]
           (let [step (pm/Slice)]
             (pm/sample 5000 :step step))))

(pyplot/pyplot
 #(az/plot_trace idata :combined true))

:bye
