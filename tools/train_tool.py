
from langchain_community.tools import tool
from langchain_core.messages import HumanMessage,SystemMessage,ToolMessage
from dotenv import load_dotenv
import os
import requests


load_dotenv()
from langchain_groq import ChatGroq
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY
)
API_KEY = os.getenv("RAPIDAPI_KEY")
HOST = os.getenv("RAPIDAPI_HOST")

RAILWAY_TOKEN=os.getenv("RAPIDAPI_TOKEN")
#tool create
@tool
def get_train_data(fromStationCode: str,
                   toStationCode: str,
                   hours: int):
    """
    Get trains between two Indian Railway station codes.

    Args:
        fromStationCode: Station code (e.g. NDLS)
        toStationCode: Station code (e.g. JP)
        hours: Number of upcoming hours to search (e.g. "24")
    """
    url = "https://irctc1.p.rapidapi.com/api/v3/getLiveStation"

    querystring = {
        "fromStationCode": fromStationCode,
        "toStationCode": toStationCode,
        "hours": str(hours)
    }

    headers = {
        "x-rapidapi-key": API_KEY,
        "x-rapidapi-host": HOST
    }

    response = requests.get(
        url,
        headers=headers,
        params=querystring,
        timeout=30
    )

    response.raise_for_status()
    return response.json()


def train_data(query: str):

    llm_with_tools = model.bind_tools([get_train_data])

    messages = [
        SystemMessage(
            content="""
You are a travel assistant.

Convert city names to station codes.

hours must ALWAYS be an integer.
"""
        ),
        HumanMessage(content=query)
    ]

    ai_msg = llm_with_tools.invoke(messages)

    messages.append(ai_msg)

    if ai_msg.tool_calls:

        tool_call = ai_msg.tool_calls[0]

        tool_result = get_train_data.invoke(tool_call["args"])

        messages.append(
            ToolMessage(content=str(tool_result),tool_call_id=tool_call["id"]))

        final = llm_with_tools.invoke(messages)

        return final.content

    return ai_msg.content


# ----------------------------
# Test
# ----------------------------
if __name__ == "__main__":

    query = "Trains from Delhi to narnaul tomorrow"

    result = train_data(query)

    print(result)


