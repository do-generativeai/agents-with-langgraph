# Import necessary libraries
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.agents import initialize_agent, tool
from langchain_community.tools import TavilySearchResults
import datetime

# Load environment variables (e.g., API keys)
load_dotenv()

# Initialize the Gemini Pro LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# Initialize the web search tool
search_tool = TavilySearchResults(search_depth="basic")

# Define a custom tool to get current system time
@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """ Returns the current date and time in the specified format """
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time

# Combine all tools into a list
tools = [search_tool, get_system_time]

# Initialize the LangChain agent with the tools and LLM
# zero-shot-react-description means the agent will decide which tool to use based on the task
agent = initialize_agent(tools=tools, llm=llm, agent="zero-shot-react-description", verbose=True)

# Run the agent with a query about cricket scores
response = agent.invoke("What is the live score of the final between India and NZ?")
print (response)