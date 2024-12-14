(ns vectorstore
(:require
 [tablecloth.api :as tc])
  (:import
   [dev.langchain4j.data.segment TextSegment]
   [dev.langchain4j.model.embedding.onnx.allminilml6v2 AllMiniLmL6V2EmbeddingModel]
   [dev.langchain4j.store.embedding.inmemory InMemoryEmbeddingStore]))

;; # Use a vector store from langchain4j
;; In this example we will create embeddings for some
;; fantasy food items, and find the closest one to a query sentence.

;; ## Create dummy data
;; First we create the data, so a list of 1000 food descriptions


(def food-items  
  ["pizza" "sushi" "ice cream" "pasta" "tacos"  
   "ramen" "paella" "croissant" "chocolate cake" "burger"])  
  

(def adjectives  
  ["delicious" "mouth-watering" "scrumptious" "heavenly" "exquisite"  
   "delectable" "savory" "luscious" "divine" "sumptuous"])  
  
(def origins  
  ["from a secret family recipe" "inspired by ancient traditions"  
   "passed down through generations" "from a hidden village"  
   "crafted by master chefs" "discovered in a remote region"  
   "with a modern twist" "from the heart of the countryside"  
   "influenced by royal cuisine" "with international flair"])  
  
(def cooking-methods  
  ["slow-cooked" "wood-fired" "pan-seared" "grilled to perfection"  
   "oven-baked" "lightly steamed" "flash-fried" "gently simmered"  
   "roasted slowly" "artisanal crafted"])  
  
(def special-ingredients  
  ["truffle-infused oil" "a hint of saffron" "exotic spices"  
   "freshly picked herbs" "a blend of rare cheeses" "organic vegetables"  
   "handmade dough" "homegrown tomatoes" "wild-caught seafood"  
   "caramelized onions"])  
  
(def effects  
  ["that delights the senses" "which melts in your mouth"  
   "that leaves you craving more" "which is a feast for the eyes"  
   "that warms the soul" "which bursts with flavor"  
   "that brings comfort and joy" "which excites the palate"  
   "that enchants every taste bud" "which is pure indulgence"])  
  
;; Generate 1000 unique descriptions as a dataset:  
(def food-descriptions
  (tc/dataset {:food-description
               (->>
                (for [food food-items]
                  (let [adj           (rand-nth adjectives)
                        origin        (rand-nth origins)
                        method        (rand-nth cooking-methods)
                        ingredient    (rand-nth special-ingredients)
                        effect        (rand-nth effects)]
                    (str "A " adj " " food ", " origin ", "
                         method " and made with " ingredient ", "
                         effect ".")))
                shuffle
                (take 1000))}))  

;; ## Add food description to vector store
;; Now we create the embedding store, which is able to calculate vector distances
;; (fast).
(def embedding-store (InMemoryEmbeddingStore.))
;; Create an instance of the embedding model, which can calculate an embedding for a piece of text.
(def embedding-model (AllMiniLmL6V2EmbeddingModel.))

;; And we embed all food descriptions:
(run!
  #(let [segment (TextSegment/from %)
         embedding (.content (.embed embedding-model %))]
     (.add embedding-store embedding segment))
  (:food-description food-descriptions))


;; Now we embed the query text:  
(def query-embedding (.content (.embed embedding-model "Which spicy food can you offer  ?")))

;; ## Query vector store
;; And finally we find the 5 most relevant embedding which are semantically the closest to the query.
;; It's using a certain vector distance (cosine) between the embedding vectors of query and texts.  
(def relevant (.findRelevant embedding-store query-embedding 5))

(tc/dataset
 (map
  (fn [a-relevant] 
    (hash-map 
     :score (.score a-relevant)
     :text (.text (.embedded a-relevant))
     )
    )
  relevant))


