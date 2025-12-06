#!/bin/bash -ex
clojure -Thfds-clj hfds-clj.models/download-cli :model '"microsoft/Phi-3-mini-4k-instruct-onnx"' :hf-token "<token>" :models-base-dir '"/tmp/models"'