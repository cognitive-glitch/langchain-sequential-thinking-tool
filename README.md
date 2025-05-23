# LangChain Sequential Thinking Tool

[![PyPI version](https://badge.fury.io/py/langchain-sequential-thinking-tool.svg)](https://badge.fury.io/py/langchain-sequential-thinking-tool) <!-- Placeholder, update after first publish -->

This package provides a LangChain `BaseTool` implementation based on the sequential thinking pattern, inspired by the Model Context Protocol (MCP) server implementation. It allows language models to structure their problem-solving process into explicit, potentially revisable thought steps.

## Features

*   **Structured Thinking:** Guides LLMs to break down problems into numbered thoughts.
*   **State Management:** Tracks thought history and allows for branching/revisions (managed within the tool instance).
*   **Rich Output:** Displays thoughts and recommendations in the console using `rich` for better visibility during agent execution.
*   **LangChain Integration:** Designed to be used seamlessly within LangChain agents.
*   **Pydantic Validation:** Uses Pydantic for robust input validation.

## Installation

```bash
pip install langchain-sequential-thinking-tool
```

## Usage

Instantiate the tool and include it in your agent's tool list. The agent's LLM should be prompted to use this tool for complex reasoning tasks, providing the required fields (`thought`, `thought_number`, `total_thoughts`, `next_thought_needed`) and optional fields (`is_revision`, `current_step`, etc.) as needed.

```python
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from sequential_thinking_tool import SequentialThinkingTool # Assuming package is installed

# 1. Initialize LLM and Tool
llm = ChatOpenAI(model="gpt-4o", temperature=0)
thinking_tool = SequentialThinkingTool()
tools = [thinking_tool]

# 2. Define Agent Prompt (example)
#    Ensure your prompt instructs the LLM on HOW and WHEN to use the sequential_thinking_tool
prompt_template = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do. For complex problems, use the sequential_thinking_tool to break down your reasoning step-by-step. Structure your thoughts clearly.
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, as a dictionary. For sequential_thinking_tool, provide 'thought', 'thought_number', 'total_thoughts', 'next_thought_needed', and optional fields like 'current_step'.
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer.
Final Answer: the final answer to the original input question

Begin!

Question: {input}
{agent_scratchpad}
"""

prompt = PromptTemplate.from_template(prompt_template)

# 3. Create Agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 4. Run the Agent
try:
    # Example task requiring sequential thought
    task = "Plan the steps to build a simple web server using Python Flask, including setting up the environment, writing basic code, and running it."
    result = agent_executor.invoke({"input": task})
    print("\nFinal Result:")
    print(result.get("output"))

    # You can inspect the tool's history after execution
    # print("\nTool History:")
    # print(json.dumps(thinking_tool.get_history(), indent=2))

except Exception as e:
    print(f"An error occurred: {e}")

```

## How it Works

The tool receives a dictionary matching the `ThoughtDataInput` schema. It validates the input, updates its internal history (stored in the `thought_history` list within the tool instance), formats the thought using `rich`, prints it to `stderr`, and returns a summary dictionary to the agent.

## Development

1.  Clone the repository.
2.  Create a virtual environment: `python -m venv .venv`
3.  Activate: `source .venv/bin/activate` (or `.\.venv\Scripts\activate` on Windows)
4.  Install dependencies: `pip install -e ".[dev]"` (Assuming a `[project.optional-dependencies]` section for dev tools like `pytest`, `build`, `twine` is added to `pyproject.toml`)
5.  Run tests: `pytest`

## Publishing to PyPI

This project uses [GitHub Actions with OIDC](https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/configuring-openid-connect-in-cloud-providers) for trusted publishing to the Python Package Index (PyPI).

The publishing process is defined in the `.github/workflows/publish-to-pypi.yml` workflow file. Creating a new [GitHub release](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository) will automatically trigger this workflow, building the package and uploading it to PyPI.

For more details on how trusted publishing works, refer to the [PyPI documentation on trusted publishers](https://docs.pypi.org/trusted-publishers/).

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

[MIT License](LICENSE)