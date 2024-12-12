package hello;


import java.util.List;
import java.util.Map;

import cljs.source_map__init;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import clojure.lang.Keyword;
import clojure.lang.PersistentHashMap;

public class Main {

    @SuppressWarnings("unchecked")
    public static Map<Keyword,Object> embedd(List<String> texts) {
        var model = new AllMiniLmL6V2EmbeddingModel();
        var store = new InMemoryEmbeddingStore<TextSegment>();

        for (String text : texts) {
            var ts = TextSegment.from(text);
            var embedding = model.embed(ts).content();
            store.add(embedding, ts);
            
        }

        
        

        
        return PersistentHashMap.create(
            Keyword.intern("store") , store,
            Keyword.intern("model") , model,
            Keyword.intern("my-texts"),texts);

    }
    
}
