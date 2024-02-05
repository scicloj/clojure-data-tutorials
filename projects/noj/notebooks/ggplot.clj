(load-file "../../header.edn")

;; # Exploring ggplot

;; Here we explore [ggplot2](https://ggplot2.tidyverse.org/) from Clojure. The goal is to eventually implement a similar grammar in Clojure.

;; ## Setup

;; We will use [ClojisR](https://scicloj.github.io/clojisr/) to call R from Clojure.

(ns ggplot
  (:require [clojisr.v1.r :as r :refer [r r$ r->clj]]
            [clojisr.v1.applications.plotting :as plotting]
            [scicloj.kindly.v4.kind :as kind]
            [clojure.walk :as walk]
            [tablecloth.api :as tc]))

(r/library "ggplot2")
(r/require-r '[base]) ; the base R library

;; ## An example plot

(-> "(ggplot(mpg, aes(cty, hwy, color=factor(cyl)))
         + geom_point()
         + stat_smooth(method=\"lm\")
         + facet_wrap(~cyl))"
    r
    plotting/plot->svg
    kind/html)

;; ## Representing plots as Clojure data

;; Inspired by
;; [cxplot](https://cxplot.com/index.html)'s internal ggplot.as.list finction, let us represent ggplot objects as R data structures.

(defn ggolot->clj
  ([r-obj
    options]
   (ggolot->clj r-obj options []))
  ([r-obj
    {:as options
     :keys [avoid]}
    path]
   #_(prn path)
   (let [relevant-names (some->> r-obj
                                 r.base/names
                                 r->clj
                                 (filter (complement avoid)))]
     (cond
       ;;
       ;; a named list or a ggproto object
       (seq relevant-names) (->> relevant-names
                                 (map (fn [nam]
                                        [(keyword nam) (-> r-obj
                                                           (r$ nam)
                                                           (ggolot->clj options
                                                                        (conj path nam)))]))
                                 (into {}))
       ;;
       ;; a ggproto method
       (-> r-obj
           r.base/class
           r->clj
           first
           (= "ggproto_method"))
       :ggproto-method
       ;;
       ;; an unnamed list
       (-> r-obj
           r.base/is-list
           r->clj
           first)
       (-> r-obj
           r.base/length
           r->clj
           first
           range
           (->> (mapv (fn [i]
                        (prn [path (inc i)])
                        (-> r-obj
                            (r/brabra (inc i))
                            (ggolot->clj options
                                         (conj path [i])))))))
       ;;
       (r.base/is-atomic r-obj) (try (r->clj r-obj)
                                     (catch Exception e
                                       (-> r-obj println with-out-str)))
       :else r-obj))))

;; For example:

(-> "(ggplot(mpg, aes(cty, hwy))
         + geom_point())"
    r
    (ggolot->clj {:avoid #{"data" "plot_env"}}))



;; ## Exlploring a few plots

;; Let us explore and compare a few plots this way:

(defn h4 [title]
  (kind/hiccup [:h3 title]))

(defn ggplot-summary [r-code]
  (let [plot (r r-code)
        clj (-> plot
                (ggolot->clj {:avoid #{"data" "plot_env"}}))]
    (kind/fragment
     [(h4 "R code")
      (kind/md
       (format "\n```{r eval=FALSE}\n%s\n```\n"
               r-code))
      (h4 "plot")
      (-> plot
          plotting/plot->buffered-image)
      (h4 "clj data")
      (kind/pprint clj)
      (h4 "portal view")
      (->> clj
           (walk/postwalk (fn [form]
                            ;; Avoiding symbols that
                            (if (or (and (symbol? form)
                                         (-> form str (= "~")))
                                    (and (keyword? form)
                                         (-> form str (= "."))))
                              (str form)
                              form)))
           kind/portal)])))

;; ### A scatterplot

(ggplot-summary
 "(ggplot(mpg, aes(cty, hwy))
         + geom_point())")

;; ### A scatterplot with colours

(ggplot-summary
 "(ggplot(mpg, aes(cty, hwy, color=factor(cyl)))
         + geom_point())")


;; ### A scatterplot with colours and smoothing

(ggplot-summary
 "(ggplot(mpg, aes(cty, hwy, color=factor(cyl)))
         + geom_point()
         + stat_smooth(method=\"lm\"))")

;; ### A scatterplot with colours, smoothing, and facets

(ggplot-summary
 "(ggplot(mpg, aes(cty, hwy, color=factor(cyl)))
         + geom_point()
         + stat_smooth(method=\"lm\")
         + facet_wrap(~cyl))")
