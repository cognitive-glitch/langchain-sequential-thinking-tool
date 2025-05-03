"""Simple tests for sequential_thinking_tool."""

from sequential_thinking_tool.tool import SequentialThinkingTool


def assert_equal(a, b, msg=""):
    if a != b:
        raise AssertionError(f"Assertion failed: {a} != {b}. {msg}")


def main():
    tool = SequentialThinkingTool()
    # initial thought
    result1 = tool.invoke(
        {
            "thought": "Test initial thought",
            "thought_number": 1,
            "total_thoughts": 2,
            "next_thought_needed": True,
        }
    )
    assert_equal(result1["thought_number"], 1, "thought_number mismatch")
    assert_equal(result1["total_thoughts"], 2, "total_thoughts mismatch")
    assert_equal(result1["next_thought_needed"], True, "next_thought_needed mismatch")
    assert_equal(result1["branches"], [], "branches should be empty")
    assert_equal(result1["thought_history_length"], 1, "history length should be 1")

    # branching thought
    result2 = tool.invoke(
        {
            "thought": "Branch from first thought",
            "thought_number": 2,
            "total_thoughts": 2,
            "next_thought_needed": True,
            "branch_from_thought": 1,
            "branch_id": "branch1",
        }
    )
    # Check branch recorded
    assert_equal(
        "branch1" in result2["branches"], True, "branch1 should be in branches"
    )
    branch_history = tool.get_branch("branch1")
    assert_equal(len(branch_history), 1, "branch history length should be 1")
    assert_equal(
        branch_history[0]["thought"],
        "Branch from first thought",
        "branch thought mismatch",
    )

    # full history
    history = tool.get_history()
    assert_equal(len(history), 2, "history length should be 2")

    # clear history
    tool.clear_history()
    assert_equal(tool.get_history(), [], "history should be empty after clear")
    assert_equal(tool.get_branch("branch1"), None, "branch should be None after clear")

    print("All tests passed")


if __name__ == "__main__":
    main()
