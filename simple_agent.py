import os
import requests
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun


# Tool 1: DuckDuckGo Search
ddg_search = DuckDuckGoSearchRun()

search_tool = Tool(
    name="DuckDuckGo Search",
    func=ddg_search.run,
    description="Search the web using DuckDuckGo and return results."
)



# Tool 2: Weather API Tool
def get_weather(city: str) -> str:
    api_key = os.getenv("WEATHER_API_KEY")
    if not api_key:
        return "ERROR: WEATHER_API_KEY is not set in environment variables."

    url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={city}"
    response = requests.get(url)

    if response.status_code != 200:
        return f"Weather API Error: {response.text}"

    data = response.json()
    location = data["location"]["name"]
    temp_c = data["current"]["temp_c"]
    condition = data["current"]["condition"]["text"]

    return f"Weather in {location}: {temp_c}°C, {condition}"


weather_tool = Tool(
    name="Weather API",
    func=get_weather,
    description="Get current weather of any city using WeatherAPI. Input should be a city name."
)


# Tool 3: Calculator Tool
def calculator(expression: str) -> str:
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"


calculator_tool = Tool(
    name="Calculator",
    func=calculator,
    description="Solve math expressions. Example input: '10*5+2'"
)


# Simple Agent Function
def run_simple_agent(query: str) -> str:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    tools = [search_tool, weather_tool, calculator_tool]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    return agent.run(query)


if __name__ == "__main__":
    user_query = input("Enter your query: ")
    print(run_simple_agent(user_query))
