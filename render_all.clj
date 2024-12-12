#!/usr/bin/env bb
(require '[babashka.process :refer [shell process exec sh]])

(def commands
  (->> "notebooks/toc.edn"
       slurp
       read-string
       (map (juxt :source-path :folder :cmd))
       (remove #(= % [nil nil]))))

(println commands)
(run!
 (fn [[source-path folder cmd]]
   (let [workspace-folder (or folder source-path)]
     (when (some? cmd)
       (shell "echo" (format "::group::{%s}" workspace-folder))
       (shell "devcontainer"  "up" "--workspace-folder" 
              workspace-folder)
       (apply shell "devcontainer"  "exec" 
              "--remote-env" (str "OPEN_AI_KEY=" (System/getenv "OPEN_AI_KEY"))       
              "--workspace-folder"  workspace-folder (str/split cmd #" "))
       (shell "echo" "::endgroup::"))))

 commands)



