# test_sequential_thinking_tool.py

import pytest
from pydantic import ValidationError

from sequential_thinking_tool.tool import SequentialThinkingTool
from sequential_thinking_tool.models import ThoughtDataInput # Import the model

# --- Helper (can be removed if using pytest's built-in assertions) ---
def assert_equal(a, b, msg=""):
    if a != b:
        raise AssertionError(f"Assertion failed: {a} != {b}. {msg}")

# --- Pydantic Validator Tests ---

def test_thought_data_input_validators():
    """Tests the custom validators in ThoughtDataInput."""
    base_valid = {
        "thought": "Test",
        "thought_number": 1,
        "total_thoughts": 1,
        "next_thought_needed": False,
    }

    # --- revises_thought validator ---
    # Valid: is_revision=True, revises_thought=1
    ThoughtDataInput(**base_valid, is_revision=True, revises_thought=1)

    # Invalid: is_revision=True, revises_thought=None
    with pytest.raises(ValidationError) as excinfo:
        ThoughtDataInput(**base_valid, is_revision=True, revises_thought=None)
    assert "revises_thought must be provided if is_revision is True" in str(excinfo.value)

    # Invalid: is_revision=False, revises_thought=1
    with pytest.raises(ValidationError) as excinfo:
        ThoughtDataInput(**base_valid, is_revision=False, revises_thought=1)
    assert "revises_thought should only be provided if is_revision is True" in str(excinfo.value)

    # Valid: is_revision=False, revises_thought=None
    ThoughtDataInput(**base_valid, is_revision=False, revises_thought=None)

    # Valid: is_revision=None, revises_thought=None (default case)
    ThoughtDataInput(**base_valid)

    # --- branch_id validator ---
    # Valid: branch_from_thought=1, branch_id="b1"
    ThoughtDataInput(**base_valid, branch_from_thought=1, branch_id="b1")

    # Invalid: branch_from_thought=1, branch_id=None
    with pytest.raises(ValidationError) as excinfo:
        ThoughtDataInput(**base_valid, branch_from_thought=1, branch_id=None)
    assert "branch_id must be provided if branch_from_thought is set" in str(excinfo.value)

    # Invalid: branch_from_thought=None, branch_id="b1"
    with pytest.raises(ValidationError) as excinfo:
        ThoughtDataInput(**base_valid, branch_from_thought=None, branch_id="b1")
    assert "branch_id should only be provided if branch_from_thought is set" in str(excinfo.value)

    # Valid: branch_from_thought=None, branch_id=None (default case)
    ThoughtDataInput(**base_valid)

# --- Tool Functionality Tests (Refactored from main) ---

@pytest.fixture
def tool():
    """Provides a fresh SequentialThinkingTool instance for each test."""
    return SequentialThinkingTool()

def test_tool_invoke_basic(tool: SequentialThinkingTool):
    """Tests basic invocation and state tracking."""
    result1 = tool.invoke(
        {
            "thought": "Test initial thought",
            "thought_number": 1,
            "total_thoughts": 2,
            "next_thought_needed": True,
        }
    )
    assert result1["thought_number"] == 1
    assert result1["total_thoughts"] == 2
    assert result1["next_thought_needed"] is True
    assert result1["branches"] == []
    assert result1["thought_history_length"] == 1

    history = tool.get_history()
    assert len(history) == 1
    assert history[0]["thought"] == "Test initial thought"

def test_tool_branching(tool: SequentialThinkingTool):
    """Tests branching functionality."""
    # Initial thought needed for branching
    tool.invoke(
        {
            "thought": "Initial",
            "thought_number": 1,
            "total_thoughts": 2,
            "next_thought_needed": True,
        }
    )
    # Branching thought
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
    assert "branch1" in result2["branches"]
    assert result2["thought_history_length"] == 2 # Main history includes branch points

    branch_history = tool.get_branch("branch1")
    assert branch_history is not None
    assert len(branch_history) == 1
    assert branch_history[0]["thought"] == "Branch from first thought"
    assert branch_history[0]["branch_id"] == "branch1"

    # Check main history contains both thoughts
    history = tool.get_history()
    assert len(history) == 2

def test_tool_history_clear(tool: SequentialThinkingTool):
    """Tests clearing the history."""
    tool.invoke({"thought": "t1", "thought_number": 1, "total_thoughts": 1, "next_thought_needed": False})
    tool.invoke({"thought": "t2", "thought_number": 1, "total_thoughts": 1, "next_thought_needed": False, "branch_from_thought": 1, "branch_id": "b1"})

    assert len(tool.get_history()) == 2
    assert tool.get_branch("b1") is not None

    tool.clear_history()

    assert tool.get_history() == []
    assert tool.get_branch("b1") is None

# --- Keep the old main() for direct execution if needed, but tests are now pytest-based ---
def main():
    """Runs the pytest tests (can be invoked directly)."""
    # This is just a convenience wrapper now. Run with `pytest` command.
    print("Running tests via pytest...")
    # You would typically run 'pytest test_sequential_thinking_tool.py' from the command line.
    # This main function doesn't execute the tests in the same way pytest does.
    # For simplicity, we'll just print a message.
    print("Please run tests using the 'pytest' command.")


if __name__ == "__main__":
    # Example of how you might run pytest programmatically, though CLI is standard.
    # pytest.main([__file__])
    main() # Keep original entry point behavior if desired, though it's less useful now.
