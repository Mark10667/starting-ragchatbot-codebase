"""
Tests for CourseSearchTool.execute method
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from unittest.mock import Mock, MagicMock
from search_tools import CourseSearchTool
from vector_store import SearchResults


class TestCourseSearchToolExecute:
    """Test CourseSearchTool.execute method"""

    def setup_method(self):
        """Set up test fixtures"""
        self.mock_store = Mock()
        self.search_tool = CourseSearchTool(self.mock_store)

    def test_execute_with_successful_results(self):
        """Test execute returns formatted results when search succeeds"""
        # Mock successful search results
        mock_results = SearchResults(
            documents=["Content about RAG systems"],
            metadata=[{"course_title": "MCP Course", "lesson_number": 1}],
            distances=[0.1],
            error=None
        )
        self.mock_store.search.return_value = mock_results
        self.mock_store.get_lesson_link.return_value = "https://example.com/lesson1"

        result = self.search_tool.execute(query="What is RAG?")

        # Verify search was called correctly
        self.mock_store.search.assert_called_once_with(
            query="What is RAG?",
            course_name=None,
            lesson_number=None
        )

        # Verify result contains content
        assert "Content about RAG systems" in result
        assert "MCP Course" in result
        print(f"✓ Test passed: execute with successful results")
        print(f"  Result: {result[:100]}...")

    def test_execute_with_empty_results(self):
        """Test execute returns appropriate message when no results found"""
        # Mock empty search results
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )
        self.mock_store.search.return_value = mock_results

        result = self.search_tool.execute(query="nonexistent topic")

        assert "No relevant content found" in result
        print(f"✓ Test passed: execute with empty results")
        print(f"  Result: {result}")

    def test_execute_with_search_error(self):
        """Test execute handles search errors correctly"""
        # Mock search error
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error="Database connection failed"
        )
        self.mock_store.search.return_value = mock_results

        result = self.search_tool.execute(query="test query")

        assert "Database connection failed" in result
        print(f"✓ Test passed: execute with search error")
        print(f"  Result: {result}")

    def test_execute_with_course_filter(self):
        """Test execute passes course_name filter correctly"""
        mock_results = SearchResults(
            documents=["Filtered content"],
            metadata=[{"course_title": "MCP Course", "lesson_number": 1}],
            distances=[0.1],
            error=None
        )
        self.mock_store.search.return_value = mock_results
        self.mock_store.get_lesson_link.return_value = None

        result = self.search_tool.execute(
            query="test",
            course_name="MCP Course"
        )

        # Verify course_name was passed to search
        self.mock_store.search.assert_called_once_with(
            query="test",
            course_name="MCP Course",
            lesson_number=None
        )
        print(f"✓ Test passed: execute with course filter")

    def test_execute_populates_last_sources(self):
        """Test execute populates last_sources for UI display"""
        mock_results = SearchResults(
            documents=["Content 1", "Content 2"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": 2}
            ],
            distances=[0.1, 0.2],
            error=None
        )
        self.mock_store.search.return_value = mock_results
        self.mock_store.get_lesson_link.side_effect = [
            "https://example.com/a/lesson1",
            "https://example.com/b/lesson2"
        ]

        result = self.search_tool.execute(query="test")

        # Verify sources were populated
        assert len(self.search_tool.last_sources) == 2
        assert self.search_tool.last_sources[0]["text"] == "Course A - Lesson 1"
        assert self.search_tool.last_sources[0]["link"] == "https://example.com/a/lesson1"
        print(f"✓ Test passed: execute populates last_sources")
        print(f"  Sources: {self.search_tool.last_sources}")


def run_tests():
    """Run all tests and report results"""
    print("\n" + "="*60)
    print("TESTING: CourseSearchTool.execute()")
    print("="*60 + "\n")

    test_class = TestCourseSearchToolExecute()
    tests = [
        test_class.test_execute_with_successful_results,
        test_class.test_execute_with_empty_results,
        test_class.test_execute_with_search_error,
        test_class.test_execute_with_course_filter,
        test_class.test_execute_populates_last_sources,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test_class.setup_method()
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"✗ Test failed: {test.__name__}")
            print(f"  Error: {str(e)}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*60}\n")

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
