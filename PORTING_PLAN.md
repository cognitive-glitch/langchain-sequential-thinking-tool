# Sequential Thinking Tool – Python Port Plan

## Overview

We will port the core sequential-thinking logic from the TypeScript MCP server to a standalone Python package that can be uploaded to PyPI and used as a LangChain BaseTool.

## Goals

1. Provide `SequentialThinkingTool` subclass of `langchain_core.tools.BaseTool`.
2. Include data models (`ThoughtData`, `ToolRecommendation`, `StepRecommendation`) in Pydantic.
3. Offer formatted output identical to TS implementation.
4. Package is PEP 517 compliant and uploadable.
5. Supply README with usage examples and CLI demonstration.

## Proposed Package Layout
```
sequential_thinking_tool/
    __init__.py
    types.py
    tool.py
pyproject.toml
README.md
```

## Mermaid Diagram
```mermaid
flowchart TB
  subgraph project root
    A[pyproject.toml] 
    B[sequential_thinking_tool/]
    C[README.md]
  end

  subgraph package "sequential_thinking_tool"
    D[__init__.py]
    E[types.py]
    F[tool.py]
  end

  A --> B
  C --> B
  B --> D & E & F
  E -->|models| E1[ThoughtData] & E2[ToolRecommendation] & E3[StepRecommendation]
  F -->|implements| F1[SequentialThinkingTool]
```

## Implementation Steps

1. **Scaffold project:**  
   - Create `pyproject.toml` using `poetry` or minimalist `[project]` table.  
   - Set dependencies: `langchain-core&nbsp;>=0.1.0`, `pydantic&nbsp;>=2.0`.

2. **Implement `types.py`:**  
   - Pydantic models replicating fields from `schema.ts` / `types.ts`.

3. **Implement `tool.py`:**  
   - Define `SequentialThinkingTool(BaseTool)` with:  
     - `name = "sequential_thinking_tool"`  
     - `description` from `schema.ts`  
     - `args_schema` referencing `ThoughtData`  
     - `_run` method implementing validation, history update, formatting.  
   - `_arun` delegates to `_run`.

4. **Implement `__init__.py`:**  
   - Export `SequentialThinkingTool`, models, `__all__`.

5. **Write `README.md`:**  
   - Installation, example usage, integration snippet with LangChain agent.

6. **Add build & publish docs:**  
   - `python -m build`, `twine upload dist/*`.

7. **Tests (optional initial):**  
   - Unit tests for `_run` validating output.

## Deliverables

- New directory `sequential_thinking_tool`.  
- Buildable `pyproject.toml`.  
- Verified upload to TestPyPI.

## Next Actions

1. Approve plan (done).  
2. Switch to Code mode to implement files.  
3. Build and validate with `pip install -e .` and `python -m pytest`.  
4. Upload to TestPyPI then PyPI.

---