(ns llms 
  (:require
   [org.httpkit.client :as hk-client]
   [cheshire.core :as json]))
 
;;# Using Large Language Models from Clojure
;;LLMs often come as APIs, as they require computing power (GPUs), which most users do not have
;;locally.
;;OpenAI offers their models behind an (paid) API for example. In the following we will see three 
;;different ways to use the GPT-4 model from OpenAI

;; Get the openai API key either from environment or a specific file
(def open-ai-key
  (or (System/getenv "OPEN_AI_KEY")
      (slurp "open_ai_secret.txt")
      )
  )


;## Use OpenAI API directly
;; OpenAI offers a rather simple API, text-in text-out for "chatting" with GPT 
;;
;; The following shows how to ask a simple question, and getting the answer using an http library,
;; [http-kit](https://github.com/http-kit/http-kit). The API is based on JSON, so easy to use
;; from Clojure


(->
 @(hk-client/post "https://api.openai.com/v1/chat/completions"
                  {:headers
                   {"content-type" "application/json"
                    "authorization" (format "Bearer %s" open-ai-key)}
                   :body
                   (json/encode
                    {:model "gpt-4"
                     :messages [{:role "system",
                                 :content "You are a helpful assistant."},
                                {:role "user",
                                 :content "What is Clojure ?"}]})})
 :body
 (json/decode keyword)) 

; ## use Bosquet
; [Bosquet](https://github.com/zmedelis/bosquet) abstracts some of the concepts of LLMs
; on a higher level API. Its has further notions of "memory" and "tools"
; and has other features we find for example in python "LangChain"

;; Bosque wants the API key in a config file
(spit "secrets.edn"
 (pr-str
  {:openai  {:api-key open-ai-key}}))


(require '[bosquet.llm.generator :refer [generate llm]])

;; Call GPT from Bosquet

(generate
 [[:user "What is Clojure"]
  [:assistant (llm :openai
                   :llm/model-params {:model :gpt-4
                                      })]])


;## Use langchain4j
;; We can use LLMs as well via a Java Interop and the library
;; [lnagchain4j](https://github.com/langchain4j/langchain4j) which aims
;; to be a copy of the python library langchain, and offers support or
;; building blocks for several concepts around LLMs (model, vector stores, document loaders, etc.)
;; We see it used in the following chapters

(import '[dev.langchain4j.model.openai OpenAiChatModel OpenAiChatModelName])

;; For now just the simplest call to an GPT model, asking it the same question:
(def open-ai-chat-model
  (.. (OpenAiChatModel/builder)
      (apiKey open-ai-key)
      (modelName OpenAiChatModelName/GPT_4)
      build))


(.generate open-ai-chat-model "What is Clojure ?")
