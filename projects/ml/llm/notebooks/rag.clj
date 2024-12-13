(ns rag
  (:require
   [clojure.java.io :as io]
   [clojure.string :as str]
   [tablecloth.api :as tc]) 
  (:import
   [dev.langchain4j.store.embedding.inmemory InMemoryEmbeddingStore]
   [dev.langchain4j.data.segment TextSegment]
   [dev.langchain4j.data.document.parser.apache.pdfbox ApachePdfBoxDocumentParser]
   [dev.langchain4j.data.document.splitter DocumentSplitters]
   [dev.langchain4j.model.embedding.onnx.allminilml6v2 AllMiniLmL6V2EmbeddingModel]))


;; # Simple RAG (Retrieval-Augmented Generation) System
;; This is a Clojure / langchain4j adaption of a
;; (simple_rag)[https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/simple_rag.ipynb]

;; ## Overview
;; This code implements a basic Retrieval-Augmented Generation (RAG) system for processing and 
;; querying PDF documents. The system encodes the document content into a vector store, 
;; which can then be queried to retrieve relevant information.

;; ## Key Components
;;PDF processing and text extraction
;;
;;Text chunking for manageable processing
;;
;;Vector store creation using InMemoryStore and AllMiniLmL6V2EmbeddingModel embeddings
;;
;;Retriever setup for querying the processed documents

;; ## Method Details
;; ### Document Preprocessing
;; The PDF is loaded using ApachePdfBox,
;; The text is split into chunks using RecursiveCharacterTextSplitter with specified chunk size and overlap.
;; ### Text Cleaning
;;A custom function replace-t-with-space is applied to clean the text chunks. 
;;This likely addresses specific formatting issues in the PDF.

;; ### Vector Store Creation
;; A AllMiniLmL6V2 embeddings are used to create vector representations of the text chunks.
;;
;; A InMemoryStore vector store is created from these embeddings for  similarity search.
;;
;; ### Retriever Setup
;; A retriever is configured to fetch the top 5 most relevant chunks for a given query.
;;
;; ## Key Features
;; Configurable Chunking: Allows adjustment of chunk size and overlap.
;;
;; Simple Retrieval: Uses InMemoryVectorStore for JVM based similarity search.
;;
;; Usage Example
;; The code includes a test query: "What is the main cause of climate change?". 
;; This demonstrates how to use the retriever to fetch relevant context from the processed document.


;; ## Benefits of this Approach
;;Scalability: Can handle large documents by processing them in chunks.
;;
;;Flexibility: Easy to adjust parameters like chunk size and number of retrieved results.

;;## Conclusion
;;This simple RAG system provides a solid foundation for building more complex information retrieval and question-answering systems. 
;;
;;By encoding document content into a searchable vector store, it enables efficient retrieval of relevant information in response to queries. 
;;
;;This approach is particularly useful for applications requiring quick access to specific information within 
;;large documents or document collections.

;; # Implementation

;; A helper to replace tabs by space:
(defn replace-t-with-space [list-of-documents]
  (map
   (fn [text-segment]
     (let [cleaned-text (-> text-segment .text (str/replace #"\t" " "))
           meta (-> text-segment .metadata)]
       (TextSegment/from cleaned-text meta)))
   list-of-documents))


;; Convert PDF to text document:
(def document (.parse (ApachePdfBoxDocumentParser.) (io/input-stream "Understanding_Climate_Change.pdf")))

;; Split document into chunks of max 1000 chars and overlaping of 200:
(def texts
  (.split 
   (DocumentSplitters/recursive 1000 200)
   document))
;; Clean texts:
(def cleaned-texts (replace-t-with-space texts))

;; Create embedding for clean texts:
(def embedding-model (AllMiniLmL6V2EmbeddingModel.))
(def embedding-store (InMemoryEmbeddingStore.))


(def embeddings
  (.embedAll embedding-model cleaned-texts))

;; Add all embeddings to vector store:
(run!
    (fn [ [text-segment embedding]]
      (.add embedding-store embedding text-segment))

 (map vector
      cleaned-texts
      (.content embeddings)))

;; Encode the retriever text:
(def retriever 
  (.content (.embed embedding-model
                    "What is the main cause of climate change?")))

;; Find top 5 relevant texts:
(def relevant (.findRelevant embedding-store retriever 5))

;; Put 5 results in table:
(tc/dataset
 (map
  (fn [a-relevant]
    (hash-map
     :score (.score a-relevant)
     :text (.text (.embedded a-relevant))))
  relevant))

