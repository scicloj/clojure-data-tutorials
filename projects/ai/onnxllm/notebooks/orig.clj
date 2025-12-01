;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
 orig
  (:require
   [uncomplicate.commons [core :refer [with-release]]]
   [uncomplicate.fluokitten.core :refer [foldmap]]
   [uncomplicate.neanderthal.core :refer [iamax transfer! native view-vctr entry!]]
   [uncomplicate.diamond
    [tensor :refer [tensor output]]
    [dnn :refer [network activation]]
    [onnxrt :refer [onnx]]]
   [uncomplicate.diamond.internal.protocols :refer [neanderthal-factory]]
   [uncomplicate.diamond.internal.onnxrt
    [core :refer [options override-dimension!]]]
   [uncomplicate.diamond.internal.dnnl.factory :refer [dnnl-factory]]
   ;[uncomplicate.diamond.internal.cudnn.factory :refer [cudnn-factory]]
   ))



(let [fact (dnnl-factory)
      neand-fact (neanderthal-factory fact)]
  (with-release [opt (-> (options)
                         (override-dimension! "batch_size" 1)
                         (override-dimension! "sequence_length" 1)
                         (override-dimension! "past_sequence_length" 1)
                         (override-dimension! "past_sequence_length + 1" 1))
                 src-tz (tensor fact [1 1 28 28] :float :nchw)
                 onnx-bp (onnx fact "/tmp/models/HuggingFaceTB/SmolLM-135M/onnx/model.onnx" {:options opt})
                 input-ids (tensor neand-fact [1 1] :long :nc)
                 position-ids (tensor neand-fact [1 1] :long :nc)
                 attention-mask (tensor neand-fact [1 1] :long :nc)
                 past-key-values (repeatedly 60 #(tensor fact [1 3 1 64] :float :nchw))
                 smollm-next! (onnx-bp (into [input-ids attention-mask position-ids] past-key-values))]
    (transfer! [2] input-ids)
    (transfer! [0] position-ids)
    (transfer! [1] attention-mask)
    (doseq [pkv past-key-values]
      (transfer! (repeat 0) pkv))
    (println
     (take 10 (view-vctr (native (first (smollm-next!))))))))
