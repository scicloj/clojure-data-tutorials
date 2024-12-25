;; # Background Removal with SVD

;; [original Fast.ai notebook](https://nbviewer.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/02-Background-Removal-with-SVD.ipynb)

(ns svd
  (:require [tablecloth.api :as tc]
            [com.phronemophobic.clj-media :as clj-media]
            [com.phronemophobic.clj-media.model :as clj-media.model]
            [tech.v3.datatype :as dtype]
            [tech.v3.tensor :as tensor]
            [tech.v3.libs.buffered-image :as bufimg]
            [scicloj.kindly.v4.kind :as kind]
            [fastmath.matrix :as mat])
  (:import (org.apache.commons.math3.linear
            SingularValueDecomposition)))

(def video-path
  "notebooks/movie/Video_003.mp4")

(kind/video
 {:src video-path})

(clj-media/probe video-path)


(def first-image
  (reduce (fn [_ frame] (clj-media.model/image
                         frame))
          nil
          (clj-media/frames
           (clj-media/file video-path)
           :video
           {:format (clj-media/video-format
                     {:pixel-format
                      :pixel-format/rgba})})))


first-image

(def first-tensor
  (bufimg/as-ubyte-tensor
   first-image))

first-tensor



(def images
  (time
   (into []
         (map clj-media.model/image)
         (clj-media/frames
          (clj-media/file video-path)
          :video
          {:format (clj-media/video-format
                    {:pixel-format
                     :pixel-format/rgba})}))))

(count images)


(def tensors
  (mapv bufimg/as-ubyte-tensor images))


(count tensors)


(def all-frames-as-one-rectangular-tensor
  (let [row-size (->> tensors
                      first
                      dtype/shape
                      (apply *))]
    (tensor/compute-tensor [row-size
                            (count tensors)]
                           (fn [j i]
                             (-> (tensors i)
                                 (tensor/reshape [row-size])
                                 (get j)))
                           :uint8)))


(def all-frames-as-one-image
  (time
   (bufimg/tensor->image
    all-frames-as-one-rectangular-tensor)))


all-frames-as-one-image


(def all-frames-as-one-matrix
  (->> all-frames-as-one-rectangular-tensor
       (take 10000)
       (map double-array)
       (mat/rows->RealMatrix)))
;; 10000x350


(def svd
  (SingularValueDecomposition.
   all-frames-as-one-matrix))

(.getSingularValues svd)
