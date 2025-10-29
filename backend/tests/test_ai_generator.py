"""
Tests for AIGenerator tool calling functionality
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from unittest.mock import Mock, MagicMock, patch
from ai_generator import AIGenerator


class MockToolManager:
    """Mock tool manager for testing"""
    def __init__(self):
        self.executed_tools = []

    def execute_tool(self, tool_name, **kwargs):
        self.executed_tools.append((tool_name, kwargs))
        return f"Mock result for {tool_name} with {kwargs}"

    def get_tool_definitions(self):
        return [{
            "name": "search_course_content",
            "description": "Search course content",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }]


class TestAIGeneratorToolCalling:
    """Test AIGenerator correctly calls tools"""

    def setup_method(self):
        """Set up test fixtures"""
        # We'll mock the Anthropic client
        self.api_key = "test-api-key"
        self.model = "claude-sonnet-4-20250514"

    @patch('ai_generator.anthropic.Anthropic')
    def test_ai_generator_detects_tool_use(self, mock_anthropic_class):
        """Test that AIGenerator detects when Claude wants to use a tool"""
        # Create mock client and response
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Mock first response with tool use
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_123"
        mock_tool_block.input = {"query": "What is RAG?"}

        mock_first_response = Mock()
        mock_first_response.stop_reason = "tool_use"
        mock_first_response.content = [mock_tool_block]

        # Mock second response after tool execution
        mock_text_block = Mock()
        mock_text_block.text = "RAG stands for Retrieval-Augmented Generation"

        mock_second_response = Mock()
        mock_second_response.content = [mock_text_block]

        mock_client.messages.create.side_effect = [
            mock_first_response,
            mock_second_response
        ]

        # Create AI generator and tool manager
        ai_gen = AIGenerator(self.api_key, self.model)
        tool_manager = MockToolManager()

        # Make request
        result = ai_gen.generate_response(
            query="What is RAG?",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )

        # Verify tool was executed
        assert len(tool_manager.executed_tools) == 1
        assert tool_manager.executed_tools[0][0] == "search_course_content"
        assert tool_manager.executed_tools[0][1]["query"] == "What is RAG?"

        # Verify we got final response
        assert "RAG stands for" in result

        # Verify two API calls were made
        assert mock_client.messages.create.call_count == 2

        print(f"✓ Test passed: AI generator detects and executes tool use")
        print(f"  Tool executed: {tool_manager.executed_tools[0]}")
        print(f"  Final response: {result}")

    @patch('ai_generator.anthropic.Anthropic')
    def test_ai_generator_handles_direct_response(self, mock_anthropic_class):
        """Test that AIGenerator handles responses without tool use"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Mock direct text response (no tool use)
        mock_text_block = Mock()
        mock_text_block.text = "General knowledge answer"

        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [mock_text_block]

        mock_client.messages.create.return_value = mock_response

        ai_gen = AIGenerator(self.api_key, self.model)
        tool_manager = MockToolManager()

        result = ai_gen.generate_response(
            query="What is 2+2?",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )

        # Verify no tools were executed
        assert len(tool_manager.executed_tools) == 0

        # Verify we got the direct response
        assert result == "General knowledge answer"

        # Verify only one API call was made
        assert mock_client.messages.create.call_count == 1

        print(f"✓ Test passed: AI generator handles direct response")
        print(f"  Response: {result}")

    @patch('ai_generator.anthropic.Anthropic')
    def test_system_prompt_includes_tool_instructions(self, mock_anthropic_class):
        """Test that system prompt mentions both tools"""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_text_block = Mock()
        mock_text_block.text = "Response"

        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [mock_text_block]

        mock_client.messages.create.return_value = mock_response

        ai_gen = AIGenerator(self.api_key, self.model)

        # Check the static system prompt
        assert "search_course_content" in ai_gen.SYSTEM_PROMPT
        assert "get_course_outline" in ai_gen.SYSTEM_PROMPT

        print(f"✓ Test passed: System prompt includes tool instructions")
        print(f"  Prompt mentions search_course_content: {'search_course_content' in ai_gen.SYSTEM_PROMPT}")
        print(f"  Prompt mentions get_course_outline: {'get_course_outline' in ai_gen.SYSTEM_PROMPT}")


def run_tests():
    """Run all tests and report results"""
    print("\n" + "="*60)
    print("TESTING: AIGenerator Tool Calling")
    print("="*60 + "\n")

    test_class = TestAIGeneratorToolCalling()
    tests = [
        test_class.test_ai_generator_detects_tool_use,
        test_class.test_ai_generator_handles_direct_response,
        test_class.test_system_prompt_includes_tool_instructions,
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
