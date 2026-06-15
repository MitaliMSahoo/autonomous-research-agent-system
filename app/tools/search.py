from langchain_community.tools import tavily_search
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults


 
@tool
def echo_modifier_tool(text: str) -> str:
    """Echo the input text."""
    return f"Echo Response: {text}"
 

def get_tavily_search_tool(max_results: int = 3) -> TavilySearchResults:
    """
    Initializes and returns the live web-browsing Tavily client.
    """
    return TavilySearchResults(max_results=max_results)



# llm = ChatAnthropic(model=setting.MODEL, temperature=0)

# # Whenever we invoke `llm_with_tool`, all three of these tool definitions
# # are passed to the model.
# tools = [my_echo_tool]
# llm_with_tools = llm.bind_tools(tools)