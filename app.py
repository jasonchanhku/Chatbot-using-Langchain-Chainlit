from langchain.agents import create_pandas_dataframe_agent
from langchain.schema import HumanMessage, SystemMessage
from langchain.agents.agent_types import AgentType
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
import pandas as pd
from dotenv import load_dotenv
import chainlit as cl
import io
import os
open_ai_key = os.getenv('OPENAI_API_KEY')

# Create an OpenAI object.
llm = ChatOpenAI(openai_api_key=open_ai_key, temperature=0)


def create_agent(data: str, llm):
    system_message = SystemMessage(content="You are a useful customer chatbot serving a user who asks questions about his Manulife insurance policies provided in the dataframe. Strictly only answer questions related to the provided dataframe")
    agent_kwargs = {
        "system_message": system_message,
    }

    # Create a Pandas DataFrame agent.
    return create_pandas_dataframe_agent(llm, data, verbose=True, agent_type=AgentType.OPENAI_FUNCTIONS,agent_kwargs=agent_kwargs)


@cl.on_chat_start
async def on_chat_start():

    await cl.Message(content="Hello there, Welcome to Customer Info Chatbot! Ask Anything about your products!").send()

    df = pd.read_csv("./mydata.csv", encoding="utf-8")

    # creating user session to store data
    cl.user_session.set('data', df)

    # Send back the response
    await cl.Message(
        content= "Data retrieved! Now you ask me anything related to your Manulife policies"
    ).send()


@cl.on_message
async def main(message: str):

    # Get data
    df = cl.user_session.get('data')

    # Agent creation
    agent = create_agent(df, llm)

    # Run model 
    response = agent.run(message)

    # Send a response back to the user
    await cl.Message(
        content=response,
    ).send()
