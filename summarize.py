import os
from dotenv import load_dotenv
from langchain.llms import Cohere
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import AnalyzeDocumentChain

load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

with open("text.txt") as f:
    text = f.read()

llm = Cohere(cohere_api_key=COHERE_API_KEY)
summary_chain = load_summarize_chain(llm, chain_type="map_reduce")
summarize_document_chain = AnalyzeDocumentChain(combine_docs_chain=summary_chain)
print(summarize_document_chain.run(text))
