(load-string (slurp  "https://raw.githubusercontent.com/scicloj/clojure-data-tutorials/main/header.edn"))

^:kindly/hide-code
(ns index)

;;# LLMs and Clojure

;;LLMs (Large Language Models) are a class of predictive models which can create "content"
;;in various forms, original "text" content.
;;
;;They are ultimately based on the completion of "text" a user is giving them. The quality of this got so good lately,
;;that we interpret this "text completions" as artificial intelligence, as they imitate with a very high quality what a human might generate.
;;
;;These models come "pre-trained", so they have learned a probability distribution of word sequences, which enables them to predict the next word
;; base on any sequence of words.
;;
;; The inner working of LLMs have as well the concept of embeddings, which means to represent text as high dimensional vectors,
;; where mathematical vector distance is correlated with semantic similarity.
;;
;; In their most popular form they are presented to users as "chat bots" with which a user can have a 
;; a coherent conversion with questions and answers.
;;
;; This being a "conversation" is an illusion from the technical level. The model itself is stateless, it uses previous parts of the conversation
;; as input for its prediction, which creates the illusion of coherence.
;;
;; The following chapters show three examples for using LLMs from Clojure:
;;
;;- a simple chat completion
;;- using a vector store and embeddings to perform a semantic search
;;- show case a simple RAG (Retrieval-Augmented Generation) use case

