import json
import sys
from typing import Optional, Type, Dict, Any, List, Tuple
from pydantic import Field

from langchain_core.tools import BaseTool, ToolException
from langchain_core.callbacks import (
    CallbackManagerForToolRun,
    AsyncCallbackManagerForToolRun,
)
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .models import ThoughtData, ThoughtDataInput, StepRecommendation, ToolRecommendation

# Replicate the description from schema.ts
TOOL_DESCRIPTION = """A detailed tool for dynamic and reflective problem-solving through thoughts.
This tool helps analyze problems through a flexible thinking process that can adapt and evolve.
Each thought can build on, question, or revise previous insights as understanding deepens.

IMPORTANT: This LangChain tool version does NOT automatically analyze available tools like the original MCP server. Tool recommendations within thoughts are based solely on the LLM's knowledge or context provided to it.

When to use this tool:
- Breaking down complex problems into steps
- Planning and design with room for revision
- Analysis that might need course correction
- Problems where the full scope might not be clear initially
- Problems that require a multi-step solution
- Tasks that need to maintain context over multiple steps
- Situations where irrelevant information needs to be filtered out

Key features:
- You can adjust total_thoughts up or down as you progress
- You can question or revise previous thoughts
- You can add more thoughts even after reaching what seemed like the end
- You can express uncertainty and explore alternative approaches
- Not every thought needs to build linearly - you can branch or backtrack
- Tracks previous recommendations and remaining steps (if provided in thoughts)

Parameters explained (within the input dictionary):
- thought: Your current thinking step.
- next_thought_needed: True if you need more thinking.
- thought_number: Current number in sequence.
- total_thoughts: Current estimate of thoughts needed.
- is_revision: (Optional) Boolean indicating if this thought revises previous thinking.
- revises_thought: (Optional) If is_revision is true, which thought number is being reconsidered.
- branch_from_thought: (Optional) If branching, which thought number is the branching point.
- branch_id: (Optional) Identifier for the current branch (if any).
- needs_more_thoughts: (Optional) If reaching end but realizing more thoughts needed.
- current_step: (Optional) Current step recommendation object.
- previous_steps: (Optional) List of previous step recommendation objects.
- remaining_steps: (Optional) List of high-level descriptions of upcoming steps.

You should:
1. Start with an initial estimate of needed thoughts, but be ready to adjust.
2. Feel free to question or revise previous thoughts.
3. Don't hesitate to add more thoughts if needed, even at the "end".
4. Express uncertainty when present.
5. Mark thoughts that revise previous thinking or branch into new paths.
6. Provide a single, ideally correct answer as the final output of the overall agent process.
7. Only set next_thought_needed to false when truly done and a satisfactory answer is reached.
"""

class SequentialThinkingTool(BaseTool):
    """
    LangChain Tool for structured, sequential thinking and problem-solving.
    """
    name: str = "sequential_thinking_tool"
    description: str = TOOL_DESCRIPTION
    args_schema: Type[ThoughtDataInput] = ThoughtDataInput
    return_direct: bool = False # Output should be processed by the agent

    # Internal state (consider thread-safety if used concurrently)
    thought_history: List[ThoughtData] = Field(default_factory=list)
    branches: Dict[str, List[ThoughtData]] = Field(default_factory=dict)
    console: Console = Field(default_factory=lambda: Console(stderr=True))

    class Config:
        # Allow Console type which isn't directly serializable
        arbitrary_types_allowed = True

    def _format_recommendation(self, step: StepRecommendation) -> Text:
        """Formats a StepRecommendation using Rich."""
        rec_text = Text()
        rec_text.append(f"Step: {step.step_description}\n", style="bold magenta")
        rec_text.append("Recommended Tools:\n", style="bold blue")
        for tool in step.recommended_tools:
            alternatives = f" (alternatives: {', '.join(tool.alternatives)})" if tool.alternatives else ''
            inputs_str = f"\n    Suggested inputs: {json.dumps(tool.suggested_inputs)}" if tool.suggested_inputs else ''
            rec_text.append(f"  - {tool.tool_name} (priority: {tool.priority}, confidence: {tool.confidence:.2f}){alternatives}\n", style="blue")
            rec_text.append(f"    Rationale: {tool.rationale}{inputs_str}\n", style="dim blue")

        rec_text.append(f"Expected Outcome: {step.expected_outcome}\n", style="bold green")
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
        """Processes a single thought step."""
        try:
            # Validate input using the Pydantic model
            validated_input = ThoughtDataInput(**kwargs)
            thought_data = ThoughtData(**validated_input.model_dump()) # Convert to internal type

            # Basic validation/adjustment
            if thought_data.thought_number > thought_data.total_thoughts:
                thought_data.total_thoughts = thought_data.thought_number # Adjust total if needed

            # --- Internal State Management ---
            # Accumulate previous steps if current step is provided
            # Note: This state management is simple; complex scenarios might need more robust handling
            if thought_data.current_step:
                # Ensure previous_steps list exists
                if not isinstance(self.thought_history[-1].previous_steps, list) if self.thought_history else False:
                     # If history exists and last entry has no list, create one
                     if self.thought_history:
                         self.thought_history[-1].previous_steps = []
                     else: # If no history, create list on current thought
                         thought_data.previous_steps = []

                # Get the list to append to (either from history or current thought)
                target_previous_steps = self.thought_history[-1].previous_steps if self.thought_history and self.thought_history[-1].previous_steps is not None else thought_data.previous_steps

                if target_previous_steps is None: # Should not happen due to above logic, but safeguard
                    target_previous_steps = []
                    if self.thought_history:
                        self.thought_history[-1].previous_steps = target_previous_steps
                    else:
                        thought_data.previous_steps = target_previous_steps

                target_previous_steps.append(thought_data.current_step)
                # If we modified history, update the current thought_data to reflect it for the return value
                if self.thought_history and self.thought_history[-1].previous_steps is target_previous_steps:
                     thought_data.previous_steps = target_previous_steps


            self.thought_history.append(thought_data)

            if thought_data.branch_from_thought and thought_data.branch_id:
                if thought_data.branch_id not in self.branches:
                    self.branches[thought_data.branch_id] = []
                self.branches[thought_data.branch_id].append(thought_data)
            # --- End State Management ---

            # Print formatted thought to stderr for visibility
            formatted_panel = self._format_thought(thought_data)
            self.console.print(formatted_panel)

            # Return structured data for the agent
            return {
                "thought_number": thought_data.thought_number,
                "total_thoughts": thought_data.total_thoughts,
                "next_thought_needed": thought_data.next_thought_needed,
                "branches": list(self.branches.keys()),
                "thought_history_length": len(self.thought_history),
                # Pass back potentially updated step info
                "current_step": thought_data.current_step.model_dump() if thought_data.current_step else None,
                "previous_steps": [step.model_dump() for step in thought_data.previous_steps] if thought_data.previous_steps else None,
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
        """Processes a single thought step asynchronously."""
        # Simple delegation for now, assuming _run is not I/O bound
        # If _run involved heavy I/O, a true async implementation would be needed
        try:
            return self._run(run_manager=run_manager.get_sync() if run_manager else None, **kwargs)
        except Exception as e:
             raise ToolException(f"Error processing thought asynchronously: {e}") from e

    def get_history(self) -> List[Dict[str, Any]]:
        """Returns the full thought history."""
        return [thought.model_dump() for thought in self.thought_history]

    def get_branch(self, branch_id: str) -> Optional[List[Dict[str, Any]]]:
        """Returns the history for a specific branch."""
        branch_history = self.branches.get(branch_id)
        return [thought.model_dump() for thought in branch_history] if branch_history else None

    def clear_history(self):
        """Clears the thought history and branches."""
        self.thought_history = []
        self.branches = {}
        self.console.print("[bold red]Thought history cleared.[/bold red]")

# Example usage (for testing purposes)
if __name__ == "__main__":
    tool = SequentialThinkingTool()

    try:
        result1 = tool.invoke({
            "thought": "Initial problem analysis: Need to port TS code to Python.",
            "thought_number": 1,
            "total_thoughts": 3,
            "next_thought_needed": True,
        })
        print("\n--- Tool Result 1 ---")
        print(json.dumps(result1, indent=2))

        result2 = tool.invoke({
            "thought": "Step 1: Define Pydantic models based on TS types.",
            "thought_number": 2,
            "total_thoughts": 3,
            "next_thought_needed": True,
             "current_step": {
                "step_description": "Define data models",
                "recommended_tools": [
                    {"tool_name": "write_file", "confidence": 0.9, "rationale": "Need to create types.py", "priority": 1, "suggested_inputs": {"path": "sequential_thinking_tool/types.py"}}
                ],
                "expected_outcome": "types.py file created with Pydantic models."
            }
        })
        print("\n--- Tool Result 2 ---")
        print(json.dumps(result2, indent=2))

        result3 = tool.invoke({
            "thought": "Step 2: Implement BaseTool subclass.",
            "thought_number": 3,
            "total_thoughts": 3,
            "next_thought_needed": False, # Assuming completion for example
            "current_step": {
                "step_description": "Implement Tool Logic",
                "recommended_tools": [
                     {"tool_name": "write_file", "confidence": 0.9, "rationale": "Need to create tool.py", "priority": 1, "suggested_inputs": {"path": "sequential_thinking_tool/tool.py"}}
                ],
                "expected_outcome": "tool.py file created with SequentialThinkingTool class.",
                "next_step_conditions": ["Test the tool locally"]
            },
            "remaining_steps": ["Write README", "Package and publish"]
        })
        print("\n--- Tool Result 3 ---")
        print(json.dumps(result3, indent=2))

        print("\n--- Full History ---")
        print(json.dumps(tool.get_history(), indent=2))

    except ToolException as e:
        print(f"\nTool Error: {e}")
    except Exception as e:
        print(f"\nGeneral Error: {e}")

    tool.clear_history()