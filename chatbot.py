from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
import pandas as pd
from langchain import PromptTemplate, LLMChain
import chainlit as cl

open_ai = "YOUR_API_KEY"

data = "McDonalds_Reviews.csv"

def create_agent(filename: str):

    # Create an OpenAI object.
    llm = OpenAI(openai_api_key=open_ai)

    # Read the CSV file into a Pandas DataFrame.
    df = pd.read_csv(filename)

    # Create a Pandas DataFrame agent.
    return create_pandas_dataframe_agent(llm, df, verbose=False)

agent = create_agent(data)

@cl.on_message
async def main(message: str):
    response = agent.run(message)

    # Send a response back to the user
    await cl.Message(
        content=response,
    ).send()
