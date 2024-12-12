(ns llms 
  (:require
   [org.httpkit.client :as hk-client]
   [cheshire.core :as json]))

;;  # using Large Language Models from Clojure
;; LLMs often come as APIs, as they require computing power (GPUs), which most users do not have
;; localy.
;; OpenAI offers their models behind an (paid) API for example. In the following we will see three 
;;diferent ways to use the GPT-4 model from OpenAI

;; get the openai API key either from environemnt or a specific file
(def open-ai-key
  (or (System/getenv "OPEN_AI_KEY")
      (slurp "open_ai_secret.txt")
      )
  )

;## Use OpenAI API directly
;; OpenAI offers a rather simple API, text-in text-out for "chatting" with GPT 
;;
;; The following shows how to ask a simple question, and getting the answer using an http libray,
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
; [bosquet](https://github.com/zmedelis/bosquet) abstracts some of the concepts of LLMs
; on a higher level API. Its has further notions of "memory" and "tools"
; and has feature we find for exampl in python "LangChain"

;; Bosque wants the API key in a config file
(spit "secrets.edn"
 (pr-str
  {:openai  {:api-key open-ai-key}}))


(require '[bosquet.llm.generator :refer [generate llm]])

(generate
 [[:user "What is Clojure"]
  [:assistant (llm :openai
                   :llm/model-params {:model :gpt-4
                                      })]])


;# use langchain4j
;; We can use LLMs as well via a Java Interop and teh library
;; [lnagchain4j](https://github.com/langchain4j/langchain4j) which aims
;; to be a copy of the pythin langcahin, and offers support or
;; build blcoks for several consept arround LLMs (model, vecstorstores, document loaders)
;; We see it used in te following chapters

(import '[dev.langchain4j.model.openai OpenAiChatModel OpenAiChatModelName])

;; For now just the simplest call to an GPT model, asking it the same question:
(def open-ai-chat-model
  (.. (OpenAiChatModel/builder)
      (apiKey open-ai-key)
      (modelName OpenAiChatModelName/GPT_4)
      build))


(.generate open-ai-chat-model "What is Clojure ?")
