import os
from dotenv import load_dotenv

from langchain.llms import Cohere
from langchain import PromptTemplate, LLMChain

load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")


def response_to_user(api_key: str, users_question: str) -> str:
    """
    Generates a response to the given user's question using the Cohere API.

    Args:
        api_key (str): The API key for the Cohere API.
        users_question (str): The question to generate a response for.

    Returns:
        str: The generated response to the user's question.
    """
    llm = Cohere(cohere_api_key=api_key)

    template = """Question:{question}
    Answer:{answer}"""

    prompt = PromptTemplate(template=template, input_variables=["question", "answer"])

    llm_chain = LLMChain(prompt=prompt, llm=llm)
    inputs = {"question": users_question, "answer": ""}
    return llm_chain.run(inputs).strip().split('\n')[0]


if __name__ == "__main__":
    questions = [
        "When is the independence day of Ukraine?",
        "Who invented the python programming language?",
        "What is the capital of France?",
    ]
    for question in questions:
        print(response_to_user(COHERE_API_KEY, question))
