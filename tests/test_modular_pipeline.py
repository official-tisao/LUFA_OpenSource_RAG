#!/usr/bin/env python3
"""
Test file for modular RAG pipeline components.
Verifies that all new modular functions work correctly.
"""

import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

# Add src directory to path and import modular components
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src import csv_utils
from src import metrics
from src import retrieval
from src import question_rewriter
from src import ground_truth
from src import answer_generator
def test_csv_utils():
    """Test CSV utility functions."""
    print("Testing CSV utilities...")

    # Create a temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("id,name,value\n1,test,0.5\n2,example,0.8\n")
        temp_path = f.name

    try:
        # Test read_csv_cached
        df = csv_utils.read_csv_cached(temp_path)
        assert len(df) == 2
        assert list(df.columns) == ['id', 'name', 'value']
        assert df.iloc[0]['name'] == 'test'

        # Test write_csv_row
        test_row = {'id': '3', 'name': 'new', 'value': '0.9'}
        result = csv_utils.write_csv_row(test_row, temp_path + '_append.csv', ['id', 'name', 'value'])
        assert result == True

        # Test get_completed_ids
        completed = csv_utils.get_completed_ids(df, 'id')
        assert '1' in completed
        assert '2' in completed

        print("      ✓ CSV utilities test passed")
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        if os.path.exists(temp_path + '_append.csv'):
            os.unlink(temp_path + '_append.csv')
def test_metrics():
    """Test metric calculation functions."""
    print("Testing metrics...")

    # Test token_f1
    f1_score = metrics.token_f1("hello world", "hello there")
    assert isinstance(f1_score, float)
    assert 0.0 <= f1_score <= 1.0

    # Test compute_bleu
    bleu_score = metrics.compute_bleu("test sentence", "test sentence for bleu")
    assert isinstance(bleu_score, float)
    assert 0.0 <= bleu_score <= 1.0

    # Test compute_rouge
    rouge_scores = metrics.compute_rouge("test sentence", "test sentence here")
    assert isinstance(rouge_scores, dict)
    assert 'rouge1' in rouge_scores
    assert 'rouge2' in rouge_scores
    assert 'rougeL' in rouge_scores

    # Test compute_meteor
    meteor_score = metrics.compute_meteor("test sentence", "test sentence here")
    assert isinstance(meteor_score, float)
    assert 0.0 <= meteor_score <= 1.0

    # Test recall_at_k
    recall = metrics.recall_at_k(['a', 'b', 'c'], ['a', 'd'], 2)
    assert isinstance(recall, float)

    # Test mrr
    mrr_score = metrics.mrr(['a', 'b', 'c'], ['a', 'd'])
    assert isinstance(mrr_score, float)

    # Test ndcg_at_k
    ndcg = metrics.ndcg_at_k(['a', 'b', 'c'], ['a', 'd'], 2)
    assert isinstance(ndcg, float)

    print("      ✓ Metrics test passed")
def test_retrieval():
    """Test retrieval module functionality."""
    print("Testing retrieval module...")

    # Create mock engine
    mock_engine = Mock()
    mock_engine._retrieve_nodes = Mock(return_value=[])
    mock_engine.similarity_top_k = 5

    # Create test DataFrame
    test_df = pd.DataFrame({
        'id': ['q1', 'q2', 'q3'],
        'question': ['What is AI?', 'How does ML work?', 'Why is Python popular?'],
        'language': ['en', 'en', 'en']
    })

    # Test retrieve_from_dataframe
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("question_id,source1_id,source1_cosine_score,source1_recency_adjusted_cosine_score,source1_rrf_score,source1_text\n")
        f.write("q1,test-id,0.85,0.85,0.85,test text\n")
        f.write("q2,test-id2,0.90,0.90,0.90,another test\n")
        temp_path = f.name

    try:
        result_df = retrieval.retrieve_from_dataframe(test_df, mock_engine, output_path=temp_path)
        assert isinstance(result_df, pd.DataFrame)
        assert 'question_id' in result_df.columns

        # Test extract_sources_from_nodes
        mock_node = Mock()
        mock_node.node.node_id = 'test-node-id'
        mock_node.score = 0.85
        mock_node.node.text = 'test text chunk'
        mock_node.node.metadata = {'original_cosine_score': '0.85', 'recency_weight': '1.0'}

        sources = retrieval.extract_sources_from_nodes([mock_node])
        assert len(sources) == 1
        assert sources[0]['node_id'] == 'test-node-id'
        assert sources[0]['cosine_score'] == 0.85
        assert sources[0]['rrf_score'] == 0.85
        assert sources[0]['text'] == 'test text chunk'

        print("      ✓ Retrieval module test passed")
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
def test_question_rewriter():
    """Test question rewriter functionality."""
    print("Testing question rewriter...")

    # Create test DataFrame
    test_df = pd.DataFrame({
        'id': ['q1', 'q2'],
        'question': ['What is artificial intelligence?', 'How machine learning works?'],
        'language': ['en', 'en']
    })

    # Mock LLM
    mock_llm = Mock()
    mock_response = Mock()
    mock_response.strip.return_value = "1. What is artificial intelligence?\n2. How machine learning algorithms work?"
    mock_llm.complete = Mock(return_value=mock_response)

    # Mock engine
    mock_engine = Mock()

    # Test batch rewrite
    result_df = question_rewriter.rewrite_questions_dataframe(
        test_df, mock_engine, mock_llm, batch_size=2
    )

    assert isinstance(result_df, pd.DataFrame)
    assert 'question_rewrite' in result_df.columns
    assert len(result_df) == 2

    # Check that rewrites were generated
    assert not result_df['question_rewrite'].isna().all()

    print("      ✓ Question rewriter test passed")
def test_answer_generator():
    """Test answer generator functionality."""
    print("Testing answer generator...")

    # Create mock engine
    mock_engine = Mock()
    mock_engine._retrieve_nodes = Mock(return_value=[])
    mock_engine._generate_from_nodes = Mock(return_value="This is a test answer")
    mock_engine.llm = Mock()
    mock_engine.similarity_top_k = 5

    # Create mock LLM for translation
    mock_llm = Mock()
    mock_llm.complete = Mock(return_value=Mock(__str__=lambda self: "This is a test answer"))

    # Test naive answer generation
    naive_result = answer_generator.naive_rag(mock_engine, "What is AI?")
    assert 'response' in naive_result
    assert naive_result['response'] == "This is a test answer"

    # Test agentic answer generation
    agentic_result = answer_generator.agentic_rag(mock_engine, "What is AI?", max_retries=2)
    assert 'response' in agentic_result
    assert agentic_result['response'] == "This is a test answer"

    print("      ✓ Answer generator test passed")
def test_ground_truth():
    """Test ground truth module functionality."""
    print("Testing ground truth...")

    # Create mock chroma data
    mock_chroma_data = {
        'ids': ['chunk1', 'chunk2'],
        'documents': ['Document about AI', 'Document about ML'],
        'metadatas': [{'article_number': '1', 'language': 'en'}]
    }

    # Create test row
    test_row = pd.Series({
        'ground_source_truth': 'Document about AI',
        'id': 'test-question',
        'question': 'What is AI?'
    })

    # Test find_ground_truth_for_row (should return empty if chroma_data not available)
    gt_ids, gt_text, alignment_pct = ground_truth.find_ground_truth_for_row(test_row, None)
    assert gt_ids == [] or isinstance(gt_ids, list)
    assert isinstance(gt_text, str)
    assert isinstance(alignment_pct, (int, float))

    print("      ✓ Ground truth test passed")
def test_modular_pipeline_integration():
    """Test integration of all modular components."""
    print("Testing modular pipeline integration...")

    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test data
        questions_df = pd.DataFrame({
            'id': ['q1', 'q2', 'q3'],
            'question': ['What is AI?', 'How ML works?', 'Why Python popular?'],
            'language': ['en', 'en', 'en'],
            'expected_answer': ['AI definition', 'ML process', 'Python reason']
        })

        # Write to temporary CSV
        questions_path = os.path.join(temp_dir, 'questions.csv')
        questions_df.to_csv(questions_path, index=False)

        # Mock components
        mock_engine = Mock()
        mock_engine._retrieve_nodes = Mock(return_value=[])
        mock_engine._generate_from_nodes = Mock(return_value="Generated answer")
        mock_engine.llm = Mock()

        mock_llm = Mock()
        mock_response = Mock()
        mock_response.strip.return_value = "Revised question"
        mock_llm.complete = Mock(return_value=mock_response)

        # Test question rewriting pipeline
        rewritten_df = question_rewriter.rewrite_questions_from_csv(
            questions_path, mock_engine, mock_llm, output_path=os.path.join(temp_dir, 'rewritten.csv')
        )

        assert isinstance(rewritten_df, pd.DataFrame)
        assert 'question_rewrite' in rewritten_df.columns

        # Test retrieval pipeline
        retrieval_df = retrieval.retrieve_from_csv_path(
            questions_path, mock_engine, output_path=os.path.join(temp_dir, 'retrieved.csv')
        )

        assert isinstance(retrieval_df, pd.DataFrame)

        print("      ✓ Modular pipeline integration test passed")
def test_csv_utility_variations():
    """Test both path-based and DataFrame-based variants of CSV utilities."""
    print("Testing CSV utility variations...")

    # Test with DataFrame version
    df = pd.DataFrame({'test': [1, 2, 3]})
    result_df = csv_utils.read_csv_cached(df)  # Should work with DataFrame
    assert isinstance(result_df, pd.DataFrame)

    # Test with path version
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("test\n1\n2\n3\n")
        temp_path = f.name

    try:
        result_df = csv_utils.read_csv_cached(temp_path)  # Should work with path
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 3
    finally:
        os.unlink(temp_path)

    print("      ✓ CSV utility variations test passed")
def run_all_tests():
    """Run all modular pipeline tests."""
    print("=" * 60)
    print("Running Modular RAG Pipeline Tests")
    print("=" * 60)

    try:
        test_csv_utils()
        test_metrics()
        test_retrieval()
        test_question_rewriter()
        test_answer_generator()
        test_ground_truth()
        test_modular_pipeline_integration()
        test_csv_utility_variations()

        print("\n" + "=" * 60)
        print("✓ All modular pipeline tests passed successfully!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)