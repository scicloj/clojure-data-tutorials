;; # Background Removal with SVD - DRAFT 🛠

;; based on: [original Fast.ai notebook](https://nbviewer.org/github/fastai/numerical-linear-algebra-v2/blob/master/nbs/02-Background-Removal-with-SVD.ipynb)

;; ## Setup

;; We use a few of the [Noj underlying libraries](https://scicloj.github.io/noj/noj_book.underlying_libraries),
;; [clj-media](https://github.com/phronmophobic/clj-media),
;; and [Apache Commons Math](https://commons.apache.org/proper/commons-math/).

(ns svd
  (:require [tablecloth.api :as tc]
            [com.phronemophobic.clj-media :as clj-media]
            [com.phronemophobic.clj-media.model :as clj-media.model]
            [tech.v3.datatype :as dtype]
            [tech.v3.tensor :as tensor]
            [tech.v3.libs.buffered-image :as bufimg]
            [scicloj.kindly.v4.kind :as kind]
            [fastmath.matrix :as mat]
            [tech.v3.datatype.functional :as dfn]
            [tech.v3.datatype.statistics :as dstats])
  (:import (org.apache.commons.math3.linear
            SingularValueDecomposition)))

;; ## Reading a video file

;; We downloaded the following file from the
;; original notebook.
;; It seems to be a shorter version of the
;; full original video (just the first 50 seconds).

(def video-path
  "notebooks/movie/Video_003.mp4")

(kind/video
 {:src video-path})

;; Let us explore it with clj-media:

(clj-media/probe video-path)

;; ## Converting the video to tensor structures

;; Using clj-media, we can reduce over frames:

(clj-media/frames
 (clj-media/file video-path)
 :video
 {:format (clj-media/video-format
           {:pixel-format
            :pixel-format/rgba})})

;; For example, let us extract the first
;; frame and convert it to an image:

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

;; When converting to a tensor, we have the four
;; color components of `rgba` format:

(bufimg/as-ubyte-tensor first-image)

;; In our case, the first component (a) is fixed:
(-> (let [t (bufimg/as-ubyte-tensor first-image)]
      (tensor/compute-tensor [240 320]
                             (fn [i j]
                               (t i j 0))
                             :uint8))
    dtype/->buffer
    distinct)

;; The rgb components are the other three.

;; We wish to process all frames, but resize
;; the images to a lower resolution, and
;; turn them to gray-scale.

;; See [Luma](https://en.wikipedia.org/wiki/Luma_(video)
;; for discussion of the gray-scale forumla:
;; 0.299 ∙ Red + 0.587 ∙ Green + 0.114 ∙ Blue

(defn image->small-tensor [image]
  (let [w 160
        h 120
        t (-> image
              (bufimg/resize w h {})
              bufimg/as-ubyte-tensor)]
    (tensor/compute-tensor [h w]
                           (fn [i j]
                             (+ (* 0.299 (t i j 1))
                                (* 0.587 (t i j 2))
                                (* 0.113 (t i j 3))))
                           :uint8)))

(-> first-image
    image->small-tensor
    bufimg/tensor->image)

;; Now let us collect the small tensors:

(def small-tensors
  (into []
        (map (comp image->small-tensor 
                   clj-media.model/image))
        (clj-media/frames
         (clj-media/file video-path)
         :video
         {:format (clj-media/video-format
                   {:pixel-format
                    :pixel-format/rgba})})))

(count small-tensors)

;; ## Reshaping the data

;; Now we will reshape the data as one matrix
;; with row per pixel and column per frame.

(def flat-tensors
  (->> small-tensors
       (mapv dtype/->buffer)))

(def long-tensor
  (tensor/compute-tensor [(-> flat-tensors first count)
                          (count flat-tensors)]
                         (fn [j i]
                           ((flat-tensors i) j))
                         :uint8))

(kind/hiccup
 [:div
  [:h3 "It is interesting to scroll this! 👇"]
  [:div {:style {:max-height "400px"
                 :overflow :auto}}
   (bufimg/tensor->image
    long-tensor)]])

;; ## Singular value decomposition

;; Let us now compute the [SVD](https://en.wikipedia.org/wiki/Singular_value_decomposition).

;; We can use Fastmath's matrix API to convert out
;; structures to the [RealMatrix](https://commons.apache.org/proper/commons-math/javadocs/api-3.6.1/org/apache/commons/math3/linear/RealMatrix.html) type of Apache Commons Math.

(def matrix
  (->> long-tensor
       (map double-array)
       (mat/rows->RealMatrix)))

(def svd
  (SingularValueDecomposition. matrix))

(.getSingularValues svd)

(def shape
  (juxt mat/nrow
        mat/ncol))

(shape (.getU svd))
(shape (.getS svd))
(shape (.getVT svd))

;; To visualize different parts of the matrix decomposition,
;; we will need to normalize tensors to the [0,1] range:
(defn tensor-normalize
  [t]
  (let [{:keys [min max]} (dstats/descriptive-statistics
                           t
                           #{:min :max})]
    (-> (dfn/- t min)
        (dfn// (- (double max) (double min))))))

;; For example:
(-> [[1 2 3]
     [4 5 6]]
    tensor/->tensor
    tensor-normalize)

;; Now let us visualize the main component of our matrix.
(def component0
  (let [i 0]
    (-> (.getColumnMatrix (.getU svd) i)
        (mat/muls (nth (.getSingularValues svd)
                       i))
        (mat/mulm (.getRowMatrix (.getVT svd) i)))))

(shape component0)

;; This is the first order approximation of the
;; pixel-by-frame matrix by the SVD method.

;; Let us take its first column, which is the first
;; frame, and show it as an image:

(defn matrix->first-image [m]
  (-> m
      (.getColumn 0)
      dtype/->array-buffer
      tensor-normalize
      (dfn/* 255)
      (dtype/->int-array)
      (tensor/reshape [120 160])
      bufimg/tensor->image))

(defn matrix->first-image [m]
  (-> m
      (.getColumn 0)
      dtype/->array-buffer
      tensor-normalize
      (dfn/* 255)
      (dtype/->int-array)
      (tensor/reshape [120 160])
      bufimg/tensor->image))

(matrix->first-image component0)

;; We see it is the background image of the video.


;; Now let us compute the remainder after removing
;; the first component.

(def residual
  (mat/sub matrix
           component0))

(matrix->first-image residual)

;; We see these are the people.


;; ## Visualizing the decomposition wit the first image:

;; Let us summarize the decomposition:

(->> [matrix
      component0
      residual]
     (mapv matrix->first-image))

;; ## Generating decomposed videos 

(defn matrix->images [m]
  (-> m
      mat/mat->array
      dtype/->array-buffer
      tensor-normalize
      (dfn/* 255)
      dtype/->int-array
      (tensor/reshape [120 160 350])
      (tensor/transpose [2 0 1])
      (->> (mapv bufimg/tensor->image))))

(->> residual
     matrix->images
     (take 20))

(def frame-format
  (clj-media/video-format {:pixel-format :pixel-format/gray8
                           :time-base 7
                           :line-size 160
                           :width 160
                           :height 120}))

(defn img->frame [img pts time-base]
  (clj-media/make-frame
   {:bytes (-> img
               (.getData)
               (.getDataBuffer)
               (.getData))
    :format frame-format
    :time-base time-base
    :pts pts}))

(def generated-frames
  (let [frame-rate 7
        seconds 50
        num-frames (* seconds frame-rate)]
    (into []
          (map-indexed (fn [pts image]
                         (img->frame image pts frame-rate)))
          (matrix->images residual))))

(def target-path
  "notebooks/generated-movie.mp4")

(clj-media/write!
 (clj-media/make-media frame-format generated-frames)
 target-path)

(kind/video
 {:src target-path})












