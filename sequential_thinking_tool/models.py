from typing import List, Optional, Dict, Any
# Import ValidationInfo
from pydantic import BaseModel, Field, field_validator, ValidationInfo

class ToolRecommendation(BaseModel):
    """Represents a recommendation for using a specific tool."""
    tool_name: str = Field(..., description="Name of the tool being recommended")
    confidence: float = Field(..., ge=0, le=1, description="0-1 indicating confidence in recommendation")
    rationale: str = Field(..., description="Why this tool is recommended")
    priority: int = Field(..., description="Order in the recommendation sequence")
    suggested_inputs: Optional[Dict[str, Any]] = Field(None, description="Optional suggested parameters")
    alternatives: Optional[List[str]] = Field(None, description="Alternative tools that could be used")

class StepRecommendation(BaseModel):
    """Represents a recommendation for a single step in the problem-solving process."""
    step_description: str = Field(..., description="What needs to be done")
    recommended_tools: List[ToolRecommendation] = Field(..., description="Tools recommended for this step")
    expected_outcome: str = Field(..., description="What to expect from this step")
    next_step_conditions: Optional[List[str]] = Field(None, description="Conditions to consider for the next step")

class ThoughtDataInput(BaseModel):
    """
    Input schema for the Sequential Thinking Tool.
    Represents a single thought or step in the thinking process.
    """
    thought: str = Field(..., description="Your current thinking step")
    thought_number: int = Field(..., description="Current thought number", ge=1)
    total_thoughts: int = Field(..., description="Estimated total thoughts needed", ge=1)
    next_thought_needed: bool = Field(..., description="Whether another thought step is needed")

    # Optional fields for revisions, branching, and recommendations
    is_revision: Optional[bool] = Field(None, description="Whether this revises previous thinking")
    revises_thought: Optional[int] = Field(None, description="Which thought is being reconsidered", ge=1)
    branch_from_thought: Optional[int] = Field(None, description="Branching point thought number", ge=1)
    branch_id: Optional[str] = Field(None, description="Branch identifier")
    needs_more_thoughts: Optional[bool] = Field(None, description="If more thoughts are needed beyond the current total")

    current_step: Optional[StepRecommendation] = Field(None, description="Current step recommendation being considered")
    previous_steps: Optional[List[StepRecommendation]] = Field(None, description="Steps already recommended")
    remaining_steps: Optional[List[str]] = Field(None, description="High-level descriptions of upcoming steps")

    # Updated validator using Pydantic v2 style
    @field_validator('revises_thought')
    @classmethod
    def check_revises_thought(cls, v: Optional[int], info: ValidationInfo) -> Optional[int]:
        # Access other fields via info.data
        is_revision = info.data.get('is_revision')
        if is_revision and v is None:
            raise ValueError('revises_thought must be provided if is_revision is True')
        if not is_revision and v is not None:
            raise ValueError('revises_thought should only be provided if is_revision is True')
        return v

    # Updated validator using Pydantic v2 style
    @field_validator('branch_id')
    @classmethod
    def check_branch_id(cls, v: Optional[str], info: ValidationInfo) -> Optional[str]:
        # Access other fields via info.data
        branch_from_thought = info.data.get('branch_from_thought')
        if branch_from_thought is not None and v is None:
            raise ValueError('branch_id must be provided if branch_from_thought is set')
        if branch_from_thought is None and v is not None:
            raise ValueError('branch_id should only be provided if branch_from_thought is set')
        return v

# This alias is for internal use within the tool's logic, mirroring the TS ThoughtData
# It includes fields that might be added/managed internally, like previous_steps accumulation.
# The input schema is ThoughtDataInput.
class ThoughtData(ThoughtDataInput):
    """Internal representation, potentially with accumulated state."""
    pass