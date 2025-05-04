import json
import sys
import threading  # Added for Lock
from typing import Any, Dict, List, Optional, Tuple, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from pydantic import Field, PrivateAttr  # Added PrivateAttr
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .models import (  # ToolRecommendation,
    StepRecommendation,
    ThoughtData,
    ThoughtDataInput,
)
from .schema import TOOL_DESCRIPTION  # Import from schema


class SequentialThinkingTool(BaseTool):
    """
    LangChain Tool for structured, sequential thinking and problem-solving.

    Attributes:
        name: The unique name of the tool.
        description: A detailed description of the tool's purpose and usage.
        args_schema: Pydantic model defining the input arguments.
        return_direct: Whether the tool's output should be returned directly to the user.
        console_kwargs: Dictionary of keyword arguments passed to the Rich Console constructor.
                        Allows configuration like output file (e.g., `{'file': open('log.txt', 'w')}`).
                        Defaults to stderr.
        verbose: If True (default), use Rich formatting for console output.
                 If False, print plain text output to the console's file handle.
    """

    name: str = "sequentialthinking_tools"  # Renamed as requested
    description: str = TOOL_DESCRIPTION
    args_schema: Type[ThoughtDataInput] = ThoughtDataInput
    return_direct: bool = False  # Output should be processed by the agent

    # --- Configurable Fields ---
    console_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments for Rich Console (e.g., file handle). Defaults to stderr.",
    )
    verbose: bool = Field(
        default=True, description="Enable/disable rich TTY formatting."
    )

    # --- Internal State (Thread-Safe) ---
    # Use PrivateAttr for internal state not part of the public API/config
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _thought_history: List[ThoughtData] = PrivateAttr(default_factory=list)
    _branches: Dict[str, List[ThoughtData]] = PrivateAttr(default_factory=dict)
    _console: Console = PrivateAttr()  # Initialized in model_post_init

    # Allow Console type which isn't directly serializable by Pydantic V1
    # For Pydantic V2, this is less critical but doesn't hurt.
    model_config = {"arbitrary_types_allowed": True}

    def model_post_init(self, __context: Any) -> None:
        """Initialize the Rich Console after Pydantic validation."""
        # Ensure console_kwargs is mutable if it came from default_factory
        resolved_kwargs = self.console_kwargs.copy()
        if "file" not in resolved_kwargs:
            resolved_kwargs["file"] = sys.stderr
        # Disable markup and highlighting if not verbose for simpler output
        if not self.verbose:
            resolved_kwargs["markup"] = False
            resolved_kwargs["highlight"] = False
            # Rich might still add some minimal ANSI codes even with no_color=True,
            # so direct print might be cleaner for truly plain output if needed.
            # However, using the console ensures output goes to the configured file.
            # resolved_kwargs["no_color"] = True # Optional: further reduce formatting

        self._console = Console(**resolved_kwargs)

    def _format_recommendation(self, step: StepRecommendation) -> Text:
        """Formats a StepRecommendation using Rich."""
        rec_text = Text()
        rec_text.append(f"Step: {step.step_description}\n", style="bold magenta")
        rec_text.append("Recommended Tools:\n", style="bold blue")
        for tool in step.recommended_tools:
            alternatives = (
                f" (alternatives: {', '.join(tool.alternatives)})"
                if tool.alternatives
                else ""
            )
            inputs_str = (
                f"\n    Suggested inputs: {json.dumps(tool.suggested_inputs)}"
                if tool.suggested_inputs
                else ""
            )
            rec_text.append(
                f"  - {tool.tool_name} (priority: {tool.priority}, confidence: {tool.confidence:.2f}){alternatives}\n",
                style="blue",
            )
            rec_text.append(
                f"    Rationale: {tool.rationale}{inputs_str}\n", style="dim blue"
            )

        rec_text.append(
            f"Expected Outcome: {step.expected_outcome}\n", style="bold green"
        )
        if step.next_step_conditions:
            rec_text.append("Conditions for next step:\n", style="bold yellow")
            for cond in step.next_step_conditions:
                rec_text.append(f"  - {cond}\n", style="yellow")
        return rec_text

    def _format_thought(self, thought_data: ThoughtData) -> Panel:
        """Formats a ThoughtData object into a Rich Panel for display."""
        prefix = ""
        context = ""
        style = "blue"

        if thought_data.is_revision:
            prefix = "🔄 Revision"
            context = f" (revising thought {thought_data.revises_thought})"
            style = "yellow"
        elif thought_data.branch_from_thought:
            prefix = "🌿 Branch"
            context = f" (from thought {thought_data.branch_from_thought}, ID: {thought_data.branch_id})"
            style = "green"
        else:
            prefix = "💭 Thought"
            context = ""
            style = "blue"

        header = f"{prefix} {thought_data.thought_number}/{thought_data.total_thoughts}{context}"
        content = Text(thought_data.thought)

        # Add recommendation information if present
        if thought_data.current_step:
            content.append("\n\n")
            content.append("Recommendation:\n", style="bold underline magenta")
            content.append(self._format_recommendation(thought_data.current_step))

        return Panel(content, title=header, border_style=style, expand=False)

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Processes a single thought step (synchronously and thread-safe)."""
        try:
            # Validate input using the Pydantic model
            validated_input = ThoughtDataInput(**kwargs)
            # Convert to internal type (which inherits from input type)
            thought_data = ThoughtData(**validated_input.model_dump())

            # --- Thread-Safe State Update ---
            with self._lock:
                # Basic validation/adjustment
                if thought_data.thought_number > thought_data.total_thoughts:
                    thought_data.total_thoughts = thought_data.thought_number

                # Accumulate previous steps if current step is provided
                # This logic assumes previous_steps are built linearly in the main history
                current_previous_steps = []
                if self._thought_history:
                    last_thought = self._thought_history[-1]
                    if last_thought.previous_steps:
                        current_previous_steps.extend(last_thought.previous_steps)
                    # Add the last thought's *current* step to the *new* thought's previous_steps
                    if last_thought.current_step:
                        current_previous_steps.append(last_thought.current_step)

                # Assign the accumulated steps to the current thought
                thought_data.previous_steps = current_previous_steps

                # Add the fully processed thought to history
                self._thought_history.append(thought_data)

                # Handle branching
                if thought_data.branch_from_thought and thought_data.branch_id:
                    if thought_data.branch_id not in self._branches:
                        self._branches[thought_data.branch_id] = []
                    self._branches[thought_data.branch_id].append(thought_data)

                # Capture state for return value *after* updates
                branches_keys = list(self._branches.keys())
                history_len = len(self._thought_history)

            # --- End Thread-Safe State Update ---

            # --- Output ---
            if self.verbose:
                formatted_panel = self._format_thought(thought_data)
                self._console.print(formatted_panel)
            else:
                # Simple text output to the configured file handle
                output_lines = [
                    f"--- Thought {thought_data.thought_number}/{thought_data.total_thoughts} ---",
                    f"Thought: {thought_data.thought}",
                ]
                if thought_data.is_revision:
                    output_lines[0] += f" (Revising: {thought_data.revises_thought})"
                elif thought_data.branch_id:
                    output_lines[
                        0
                    ] += f" (Branch: {thought_data.branch_id} from {thought_data.branch_from_thought})"

                if thought_data.current_step:
                    output_lines.append(
                        f"Recommendation: {thought_data.current_step.step_description}"
                    )
                    output_lines.append(
                        f"  Expected Outcome: {thought_data.current_step.expected_outcome}"
                    )
                    # Add more details if needed for non-verbose

                # Use console's file handle directly for output destination
                output_str = "\n".join(output_lines)
                print(
                    output_str, file=self._console.file, flush=True
                )  # Ensure output is written

            # --- End Output ---

            # Return structured data for the agent
            return {
                "thought_number": thought_data.thought_number,
                "total_thoughts": thought_data.total_thoughts,
                "next_thought_needed": thought_data.next_thought_needed,
                "branches": branches_keys,  # Use state captured after lock release
                "thought_history_length": history_len,  # Use state captured after lock release
                # Pass back potentially updated step info
                "current_step": (
                    thought_data.current_step.model_dump()
                    if thought_data.current_step
                    else None
                ),
                "previous_steps": (
                    [step.model_dump() for step in thought_data.previous_steps]
                    if thought_data.previous_steps
                    else None
                ),
                "remaining_steps": thought_data.remaining_steps,
            }

        except Exception as e:
            # Use ToolException for errors during tool execution
            raise ToolException(f"Error processing thought: {e}") from e

    async def _arun(
        self,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """This tool is synchronous only due to internal state and Rich printing."""
        raise NotImplementedError(
            "SequentialThinkingTool does not support asynchronous execution."
        )

    def get_history(self) -> List[Dict[str, Any]]:
        """Returns the full thought history (thread-safe)."""
        with self._lock:
            # model_dump creates copies, ensuring thread safety of the returned data
            return [thought.model_dump() for thought in self._thought_history]

    def get_branch(self, branch_id: str) -> Optional[List[Dict[str, Any]]]:
        """Returns the history for a specific branch (thread-safe)."""
        with self._lock:
            branch_history = self._branches.get(branch_id)
            # model_dump creates copies
            return (
                [thought.model_dump() for thought in branch_history]
                if branch_history
                else None
            )

    def clear_history(self):
        """Clears the thought history and branches (thread-safe)."""
        with self._lock:
            self._thought_history = []
            self._branches = {}
        # Print confirmation outside the lock
        if self.verbose:
            self._console.print("[bold red]Thought history cleared.[/bold red]")
        else:
            # Use console's file handle directly
            print("Thought history cleared.", file=self._console.file, flush=True)


# Example usage (for testing purposes)
if __name__ == "__main__":
    # Example: Writing non-verbose output to a file
    # with open("thought_log.txt", "w", encoding="utf-8") as f:
    #     tool = SequentialThinkingTool(verbose=False, console_kwargs={"file": f, "width": 120})

    # Default verbose output to stderr
    tool = SequentialThinkingTool()

    print("\n--- Running Tool Examples ---")

    try:
        result1 = tool.invoke(
            {
                "thought": "Initial problem analysis: Need to port TS code to Python.",
                "thought_number": 1,
                "total_thoughts": 3,
                "next_thought_needed": True,
            }
        )
        print("\n--- Tool Result 1 (JSON) ---")
        print(json.dumps(result1, indent=2))

        result2 = tool.invoke(
            {
                "thought": "Step 1: Define Pydantic models based on TS types.",
                "thought_number": 2,
                "total_thoughts": 3,
                "next_thought_needed": True,
                "current_step": {
                    "step_description": "Define data models",
                    "recommended_tools": [
                        {
                            "tool_name": "write_file",
                            "confidence": 0.9,
                            "rationale": "Need to create types.py",
                            "priority": 1,
                            "suggested_inputs": {
                                "path": "sequential_thinking_tool/types.py"
                            },
                        }
                    ],
                    "expected_outcome": "types.py file created with Pydantic models.",
                },
            }
        )
        print("\n--- Tool Result 2 (JSON) ---")
        print(json.dumps(result2, indent=2))

        result3 = tool.invoke(
            {
                "thought": "Step 2: Implement BaseTool subclass.",
                "thought_number": 3,
                "total_thoughts": 3,
                "next_thought_needed": False,  # Assuming completion for example
                "current_step": {
                    "step_description": "Implement Tool Logic",
                    "recommended_tools": [
                        {
                            "tool_name": "write_file",
                            "confidence": 0.9,
                            "rationale": "Need to create tool.py",
                            "priority": 1,
                            "suggested_inputs": {
                                "path": "sequential_thinking_tool/tool.py"
                            },
                        }
                    ],
                    "expected_outcome": "tool.py file created with SequentialThinkingTool class.",
                    "next_step_conditions": ["Test the tool locally"],
                },
                "remaining_steps": ["Write README", "Package and publish"],
            }
        )
        print("\n--- Tool Result 3 (JSON) ---")
        print(json.dumps(result3, indent=2))

        print("\n--- Full History (JSON) ---")
        print(json.dumps(tool.get_history(), indent=2))

    except ToolException as e:
        print(f"\nTool Error: {e}", file=sys.stderr)
    except Exception as e:
        print(f"\nGeneral Error: {e}", file=sys.stderr)
    finally:
        # Ensure the file handle is closed if opened in the example
        if "file" in tool.console_kwargs and hasattr(
            tool.console_kwargs["file"], "close"
        ):
            # Check if it's not sys.stderr/stdout before closing
            if tool.console_kwargs["file"] not in (sys.stderr, sys.stdout):
                try:
                    tool.console_kwargs["file"].close()
                    print(f"\nClosed file: {tool.console_kwargs['file'].name}")
                except Exception as close_err:
                    print(f"\nError closing file: {close_err}", file=sys.stderr)

        tool.clear_history()
        print("\n--- Tool Examples Finished ---")
