
        result = {
            'response':          str(response),
            'detected_language': detected_language,
        }

        if return_sources and hasattr(response, 'source_nodes'):
            sources = []
            for node in response.source_nodes:
                text    = node.node.text
                preview = text[:PREVIEW_LENGTH] + ('...' if len(text) > PREVIEW_LENGTH else '')
                sources.append({
                    'text':     preview,
                    'score':    node.score,
                    'metadata': node.node.metadata,
                    'node_id':  node.node.node_id
                })
            result['sources'] = sources

        return result

    def chat(self, query_text: str) -> str:
        return self.query(query_text, return_sources=False)['response']

    def _retrieve_nodes(self, query: str, top_k: int = 5) -> list:
        """
        Executes advanced hybrid retrieval. Sorts dense vector nodes by recency weights
        during similarity ties, then merges them with sparse BM25 tokens via RRF.
        Directly reads from ChromaDB to avoid ZeroDivisionError.
        """
        dense_retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=top_k * 2,
        )
        raw_dense_nodes = dense_retriever.retrieve(query)

        if raw_dense_nodes:
            raw_dense_nodes.sort(key=lambda x: x.score, reverse=True)
            dense_nodes = []
            i = 0
            tie_threshold = 0.02

            while i < len(raw_dense_nodes):
                bucket_score = raw_dense_nodes[i].score
                bucket = []
                j = i
                while j < len(raw_dense_nodes) and (bucket_score - raw_dense_nodes[j].score) < tie_threshold:
                    bucket.append(raw_dense_nodes[j])
                    j += 1

                bucket.sort(key=lambda x: float(x.node.metadata.get("recency_weight", 1.0)), reverse=True)
                dense_nodes.extend(bucket)
                i = j
        else:
            dense_nodes = []

        from llama_index.core.schema import TextNode
        chroma_client = chromadb.PersistentClient(path=self.db_path)
        chroma_collection = chroma_client.get_or_create_collection(self.collection_name)
        collection_data = chroma_collection.get(include=["documents", "metadatas"])

        all_nodes = []
        if collection_data and collection_data.get("ids"):
            for cid, doc, meta in zip(collection_data["ids"], collection_data["documents"], collection_data["metadatas"]):
                node = TextNode(id_=cid, text=doc, metadata=meta)
                all_nodes.append(node)

        if not all_nodes:
            return dense_nodes

        tokenized_corpus = [re.findall(r'\b\w+\b', node.text.lower()) for node in all_nodes]
        bm25 = BM25Okapi(tokenized_corpus)

        tokenized_query = re.findall(r'\b\w+\b', query.lower())
        bm25_scores = bm25.get_scores(tokenized_query)

        top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda idx: bm25_scores[idx], reverse=True)[:top_k * 2]
        sparse_nodes = [all_nodes[idx] for idx in top_bm25_indices]

        rrf_scores = {}
        node_mapping = {}

        for rank, node in enumerate(dense_nodes, start=1):
            node_id = node.node.node_id
            node_mapping[node_id] = node
            node.node.metadata["original_cosine_score"] = str(node.score)
            rrf_scores[node_id] = rrf_scores.get(node_id, 0.0) + (1.0 / (60 + rank))

        for rank, node in enumerate(sparse_nodes, start=1):
            node_id = node.node_id
            if node_id not in node_mapping:
                from llama_index.core.schema import NodeWithScore
                node_mapping[node_id] = NodeWithScore(node=node, score=0.0)
            rrf_scores[node_id] = rrf_scores.get(node_id, 0.0) + (1.0 / (60 + rank))

        sorted_node_ids = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)[:top_k]

        final_nodes = []
        for nid in sorted_node_ids:
            wrapped_node = node_mapping[nid]
            wrapped_node.score = rrf_scores[nid]
            final_nodes.append(wrapped_node)

        return final_nodes

    def _generate_from_nodes(self, original_query: str, nodes, lang: str) -> str:
