from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
import os

load_dotenv()

information = """Dr. Dolittle is a fictional physician who can talk to animals. He prefers the company of animals to humans and lives with many pets. 
He is known for his kind heart and willingness to help any creature in need. His ability to communicate with animals makes him uniquely suited to treat their ailments.
The doctor is generally quiet and reserved around people but becomes quite chatty and animated when speaking with his animal friends."""

if __name__ == "__main__":
    load_dotenv()
    
    summary_template = """
    Given the information {information} about a person from I want you to create:
    1. a short summary
    2. a list of 2-3 key details

    Is the person extroverted, introverted, empathetic, competitive, or something else?
    """


    summary_prompt_template = PromptTemplate(
        input_variables="information", template=summary_template
    )

    llm = ChatAnthropic(model_name="claude-3-5-sonnet-20240620", temperature=0)

    chain = summary_prompt_template | llm

    res = chain.invoke(input={"information": information})

    print(res.content)
