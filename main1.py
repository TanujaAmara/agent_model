from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun

from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_classic import hub


# Load environment variables
load_dotenv()


# -------------------------------
# Shared Tools for all agents
# -------------------------------
web_search_tool = DuckDuckGoSearchRun()

wikipedia_tool = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=700)
)

arxiv_tool = ArxivQueryRun()

tools = [web_search_tool, wikipedia_tool, arxiv_tool]


# -------------------------------
# Helper: Build Agent Executor
# -------------------------------
def build_agent_executor():
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0
    )

    prompt = hub.pull("hwchase17/react")

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True
    )

    return executor


# -------------------------------
# Agent 1: Research Agent
# -------------------------------
def research_agent(query: str) -> str:
    executor = build_agent_executor()
    output = executor.invoke({"input": f"Research this topic with key points:\n{query}"})
    return output["output"]


# -------------------------------
# Agent 2: Summarizer Agent
# -------------------------------
def summarizer_agent(research_text: str) -> str:
    executor = build_agent_executor()
    output = executor.invoke({"input": f"Summarize this into short bullet points:\n\n{research_text}"})
    return output["output"]


# -------------------------------
# Agent 3: Email Agent
# -------------------------------
def email_agent(summary_text: str) -> str:
    executor = build_agent_executor()
    output = executor.invoke(
        {
            "input": f"""
Write a professional email using this summary.

Must include:
- Subject
- Greeting
- Body
- Closing

Summary:
{summary_text}
"""
        }
    )
    return output["output"]


# -------------------------------
# Main Pipeline (Orchestrator)
# -------------------------------
def run_main_pipeline(query: str) -> dict:
    research_output = research_agent(query)
    summary_output = summarizer_agent(research_output)
    email_output = email_agent(summary_output)

    return {
        "Research Output": research_output,
        "Summary Output": summary_output,
        "Email Output": email_output
    }


# -------------------------------
# Test Run
# -------------------------------
if __name__ == "__main__":
    topic = input("Enter your topic: ")
    result = run_main_pipeline(topic)

    print("\n====== Research Output ======\n")
    print(result["Research Output"])

    print("\n====== Summary Output ======\n")
    print(result["Summary Output"])

    print("\n====== Email Output ======\n")
    print(result["Email Output"])
