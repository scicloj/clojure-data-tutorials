# Contributing 


## Add your own folder

1. Create a folder in `projects` with a [Clay](https://scicloj.github.io/clay/) notebook, multiple notebooks, or a book. Add a link to the scrapbook at the top of your notebooks. 
You need to have a `.devcontainer` folder in it.

2. Create a [Pull Request](https://github.com/scicloj/clojure-data-tutorials/pulls) adding our new folder at the bottom of the [table of contents](https://github.com/scicloj/clojure-data-tutorials/blob/main/notebooks/toc.edn).

You need to specify 
| tag | example | description|
|-----|--------| ------- |
|:created       | "2024-01-11"| creadion date of notebook|
|:title         | "Machine learning - DRAFT"|title of notebook|
|:url           |  "projects/noj/ml.html"|  entry point html file of the notebook|
|:source-path   | "projects/noj/notebooks/ml.clj"| folder or file as entrypoint of the notebook|
|:folder        |  "projects/noj" | base folder, not needed if :souce path is a folder|
|:cmd           | "clj notebooks/render.clj" | command to render notebook|
|:tags          | [:noj :ml :scicloj.ml :draft] | some tags|

If `:source-path` is a folder, `:folder` is not needed
(A clay notebook can be ither a .clj fil or a folder of .clj files)

`:cmd` need to be a command, which renders your notebook into `docs/` folder


