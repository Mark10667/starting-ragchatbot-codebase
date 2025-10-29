"""
Run all test suites and provide comprehensive report
"""
# -*- coding: utf-8 -*-
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import test_search_tool
import test_ai_generator
import test_rag_system_integration


def main():
    """Run all test suites"""
    print("\n" + "="*70)
    print(" RAG CHATBOT DIAGNOSTIC TEST SUITE")
    print("="*70)

    all_passed = True

    # Run search tool tests
    print("\n[1/3] Running CourseSearchTool tests...")
    search_tool_passed = test_search_tool.run_tests()
    all_passed = all_passed and search_tool_passed

    # Run AI generator tests
    print("\n[2/3] Running AIGenerator tests...")
    ai_gen_passed = test_ai_generator.run_tests()
    all_passed = all_passed and ai_gen_passed

    # Run RAG system integration tests
    print("\n[3/3] Running RAG System integration tests...")
    rag_system_passed = test_rag_system_integration.run_tests()
    all_passed = all_passed and rag_system_passed

    # Final summary
    print("\n" + "="*70)
    print(" FINAL TEST SUMMARY")
    print("="*70)
    print(f"CourseSearchTool tests:  {'✓ PASSED' if search_tool_passed else '✗ FAILED'}")
    print(f"AIGenerator tests:       {'✓ PASSED' if ai_gen_passed else '✗ FAILED'}")
    print(f"RAG System tests:        {'✓ PASSED' if rag_system_passed else '✗ FAILED'}")
    print("="*70)

    if all_passed:
        print("\n✓ ALL TESTS PASSED - System appears to be working correctly")
        print("  If you're still seeing 'query failed', the issue may be:")
        print("  1. API key configuration")
        print("  2. Database not populated with course data")
        print("  3. Network/API connectivity issues")
    else:
        print("\n✗ SOME TESTS FAILED - See details above")
        print("  Review the failed tests to identify the root cause")

    print("\n")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
