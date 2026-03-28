import os
from dotenv import load_dotenv

from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities import WikipediaAPIWrapper

# Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("API key not found. Check your .env file.")

# LLM
llm = ChatOpenAI(
    temperature=0,
    model="gpt-4o-mini"
)

# Tools
search = DuckDuckGoSearchRun()
wiki = WikipediaAPIWrapper()

tools = [
    Tool(
        name="Web Search",
        func=search.run,
        description="Search latest information from the internet"
    ),
    Tool(
        name="Wikipedia",
        func=wiki.run,
        description="Get factual and general knowledge"
    ),
]

# Agent (ReAct)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)


def generate_report(topic):
    prompt = f"""
You are an AI research agent.

Create a structured research report on: {topic}

STRICT FORMAT:

1. Cover Page
   - Title
   - Date

2. Title

3. Introduction

4. Key Findings (use bullet points)

5. Challenges

6. Future Scope

7. Conclusion

Ensure:
- Clear headings
- Professional tone
- Real-world examples where possible
"""

    return agent.run(prompt)


if __name__ == "__main__":
    topic = input("Enter topic: ")
    report = generate_report(topic)

    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print("\n\n===== FINAL REPORT =====\n")
    print(report)