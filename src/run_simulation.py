
            lang = rag_engine.detect_query_language(q_text)
            nodes = rag_engine._retrieve_nodes(q_text, top_k=5)
            answer = copilot.generate_from_nodes(q_text, nodes, lang)

            sources_list = []
            for n in nodes:
                combined_meta = {}
                for k, v in n.node.metadata.items():
                    combined_meta[k] = v
                combined_meta["id"] = n.node.node_id
                if "original_cosine_score" in n.node.metadata:
                    combined_meta["original_cosine_score"] = n.node.metadata["original_cosine_score"]
                else:
                    combined_meta["original_cosine_score"] = str(n.score)

                sources_list.append({
                    "text": n.node.text[:500],
                    "score": n.score,
                    "metadata": combined_meta,
                    "node_id": n.node.node_id
                })

            result = {
                "response": answer,
                "original_language": record.get("language", lang),
                "rewritten_query": "",
                "attempts": 1,
                "grounded": True,
                "sources": sources_list
            }

        sources = result.get("sources", [])
        sources_dict = extract_sources(sources)

        orig_cosine = ""
        recency_adj = ""
        rrf_val = ""
        if sources:
            first_src = sources[0]
            first_meta = first_src.get("metadata", {})
            rrf_val = round(float(first_src.get("score") or 0.0), 6)
            orig_cosine = round(float(first_meta.get("original_cosine_score") or rrf_val), 6)
            recency_adj = orig_cosine

        row = {
            "question_id": q_id,
            "question": q_text,
            "answer": result.get("response", ""),
            "base_model_used": model_name,
            "language": result.get("original_language", record.get("language", "en")),
            "attempts": result.get("attempts", 1),
            "grounded": result.get("grounded", False),
        }
        row.update(sources_dict)
        row["original_cosine_score"] = orig_cosine
        row["recency_adjusted_score"] = recency_adj
        row["RRF"] = rrf_val

        print(f"   [Simulation Engine] ✅ Success! Received Answer length: {len(row['answer'])} chars.")
        return row

    except Exception as e:
        print(f"   [Simulation Engine] 💥 Error on record {q_id}: {e}")
        traceback.print_exc()
        return _empty_row(q_id, q_text, model_name, record.get("language", "en"))


def _empty_row(q_id, q_text, model, lang):
    row = {
        "question_id": q_id,
        "question": q_text,
        "answer": "ERROR",
        "base_model_used": model,
        "language": lang,
        "attempts": 0,
        "grounded": False,
    }
    for i in range(1, 6):
        row[f"source{i}_id"] = ""
        row[f"source{i}_score"] = ""
        row[f"source{i}_text"] = ""
    row["original_cosine_score"] = ""
    row["recency_adjusted_score"] = ""
    row["RRF"] = ""
    return row


def ensure_ground_truth(csv_path):
    """Auto-run find_ground_truth.py if ground_source_truth_id column is missing."""
    df = pd.read_csv(csv_path)
    if "ground_source_truth_id" not in df.columns or df["ground_source_truth_id"].isnull().all():
        print("[Sim] ground_source_truth_id missing — running find_ground_truth.py first...")
        from find_ground_truth import run as find_gt
        find_gt(csv_path, "db/chroma_db", "multilingual_docs", top_k=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LUFA RAG simulation over test dataset")
    parser.add_argument("--mode", choices=["local", "api", "frontier"], default="local")
