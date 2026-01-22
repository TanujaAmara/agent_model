import os
import requests
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_classic.agents import hub


# Load environment variables from .env
load_dotenv()


# -------------------------------
# TOOL 1: DuckDuckGo Search Tool
# -------------------------------
ddg_search = DuckDuckGoSearchRun()


# -------------------------------
# TOOL 2: Weather API Tool
# -------------------------------
@tool
def weather_api(city: str) -> str:
    """Get current weather for a given city using WeatherAPI."""
    api_key = os.getenv("WEATHER_API_KEY")

    if not api_key:
        return "ERROR: WEATHER_API_KEY not found. Please set it in .env file."

    url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={city}"
    response = requests.get(url)

    if response.status_code != 200:
        return f"Weather API error: {response.text}"

    data = response.json()
    location = data["location"]["name"]
    temp_c = data["current"]["temp_c"]
    condition = data["current"]["condition"]["text"]

    return f"Weather in {location}: {temp_c}°C, {condition}"


# -------------------------------
# TOOL 3: Calculator Tool
# -------------------------------
@tool
def calculator(expression: str) -> str:
    """Solve a math expression. Example: '10*5+2'."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation Error: {str(e)}"


# -------------------------------
# Build Simple Agent
# -------------------------------
def build_simple_agent():
    # ✅ Gemini (Google API Key)
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0
    )

    tools = [ddg_search, weather_api, calculator]

    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return executor


# -------------------------------
# Run Simple Agent
# -------------------------------
def run_simple_agent(query: str) -> str:
    executor = build_simple_agent()
    result = executor.invoke({"input": query})
    return result["output"]


# -------------------------------
# Test Run
# -------------------------------
if __name__ == "__main__":
    user_query = input("Enter your question: ")
    print(run_simple_agent(user_query))
