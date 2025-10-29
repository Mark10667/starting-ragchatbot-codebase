"""
Integration tests for RAG system handling content queries
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from unittest.mock import Mock, patch, MagicMock
from rag_system import RAGSystem
from vector_store import SearchResults
import tempfile
import shutil


class MockConfig:
    """Mock configuration for testing"""
    def __init__(self):
        self.ANTHROPIC_API_KEY = "test-key"
        self.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
        self.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        self.CHUNK_SIZE = 800
        self.CHUNK_OVERLAP = 100
        self.MAX_RESULTS = 5
        self.MAX_HISTORY = 2
        # Use a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.CHROMA_PATH = os.path.join(self.temp_dir, "test_chroma")

    def cleanup(self):
        """Clean up temporary directory"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


class TestRAGSystemContentQueries:
    """Test RAG system handling of content-related questions"""

    def setup_method(self):
        """Set up test fixtures"""
        self.config = MockConfig()

    def teardown_method(self):
        """Clean up after tests"""
        self.config.cleanup()

    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    def test_rag_system_passes_tools_to_ai(self, mock_vector_store_class, mock_ai_gen_class):
        """Test that RAG system passes tool definitions to AI generator"""
        # Mock the AI generator
        mock_ai_gen = Mock()
        mock_ai_gen.generate_response.return_value = "Test response"
        mock_ai_gen_class.return_value = mock_ai_gen

        # Mock the vector store
        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store

        # Create RAG system
        rag = RAGSystem(self.config)

        # Make a query
        response, sources = rag.query("What is RAG?")

        # Verify AI generator was called with tools
        mock_ai_gen.generate_response.assert_called_once()
        call_kwargs = mock_ai_gen.generate_response.call_args[1]

        assert 'tools' in call_kwargs, "Tools not passed to AI generator"
        assert 'tool_manager' in call_kwargs, "Tool manager not passed to AI generator"
        assert call_kwargs['tools'] is not None, "Tools list is None"

        # Verify tools include search_course_content
        tools = call_kwargs['tools']
        tool_names = [tool['name'] for tool in tools]
        assert 'search_course_content' in tool_names, "search_course_content tool not found"
        assert 'get_course_outline' in tool_names, "get_course_outline tool not found"

        print(f"✓ Test passed: RAG system passes tools to AI")
        print(f"  Available tools: {tool_names}")

    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    def test_rag_system_retrieves_sources_from_tool(self, mock_vector_store_class, mock_ai_gen_class):
        """Test that RAG system retrieves sources from tool manager after query"""
        # Mock the AI generator
        mock_ai_gen = Mock()
        mock_ai_gen.generate_response.return_value = "RAG is a technique..."
        mock_ai_gen_class.return_value = mock_ai_gen

        # Mock the vector store
        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store

        # Create RAG system
        rag = RAGSystem(self.config)

        # Manually set sources in the search tool (simulating tool execution)
        rag.search_tool.last_sources = [
            {"text": "MCP Course - Lesson 1", "link": "https://example.com/lesson1"}
        ]

        # Make a query
        response, sources = rag.query("What is RAG?")

        # Verify sources were retrieved
        assert len(sources) == 1
        assert sources[0]["text"] == "MCP Course - Lesson 1"

        print(f"✓ Test passed: RAG system retrieves sources")
        print(f"  Sources: {sources}")

    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    def test_rag_system_resets_sources_after_query(self, mock_vector_store_class, mock_ai_gen_class):
        """Test that RAG system resets sources after retrieving them"""
        # Mock the AI generator
        mock_ai_gen = Mock()
        mock_ai_gen.generate_response.return_value = "Response"
        mock_ai_gen_class.return_value = mock_ai_gen

        # Mock the vector store
        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store

        # Create RAG system
        rag = RAGSystem(self.config)

        # Set sources
        rag.search_tool.last_sources = [{"text": "Source 1", "link": None}]

        # Make a query
        response, sources = rag.query("Test query")

        # Verify sources were reset
        assert len(rag.search_tool.last_sources) == 0, "Sources not reset after query"

        print(f"✓ Test passed: RAG system resets sources after query")

    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    def test_rag_system_query_format(self, mock_vector_store_class, mock_ai_gen_class):
        """Test that RAG system formats the query correctly"""
        # Mock the AI generator
        mock_ai_gen = Mock()
        mock_ai_gen.generate_response.return_value = "Response"
        mock_ai_gen_class.return_value = mock_ai_gen

        # Mock the vector store
        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store

        # Create RAG system
        rag = RAGSystem(self.config)

        # Make a query
        user_query = "What is RAG?"
        response, sources = rag.query(user_query)

        # Verify the query was formatted
        call_kwargs = mock_ai_gen.generate_response.call_args[1]
        query_sent = call_kwargs['query']

        assert "Answer this question about course materials:" in query_sent
        assert user_query in query_sent

        print(f"✓ Test passed: RAG system formats query correctly")
        print(f"  Formatted query: {query_sent}")


def run_tests():
    """Run all tests and report results"""
    print("\n" + "="*60)
    print("TESTING: RAG System Content Query Handling")
    print("="*60 + "\n")

    test_class = TestRAGSystemContentQueries()
    tests = [
        test_class.test_rag_system_passes_tools_to_ai,
        test_class.test_rag_system_retrieves_sources_from_tool,
        test_class.test_rag_system_resets_sources_after_query,
        test_class.test_rag_system_query_format,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test_class.setup_method()
            test()
            test_class.teardown_method()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"✗ Test failed: {test.__name__}")
            print(f"  Error: {str(e)}")
            import traceback
            traceback.print_exc()
            test_class.teardown_method()

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*60}\n")

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
