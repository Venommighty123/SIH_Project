from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.document_loaders import CSVLoader
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

model = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash-lite",
    api_key = api_key,
    temperature = 0.5,
    max_tokens = 512
)

class suggestion(BaseModel):
    fact_1: str = Field(default = None, description = "Suggestion 1 based upon the prompt")
    fact_2: str = Field(default = None, description = "Suggestion 2 based upon the prompt")
    fact_3: str = Field(default = None, description = "Suggestion 3 based upon the prompt")
    fact_4: str = Field(default = None, description = "Suggestion 4 based upon the prompt")
    fact_5: str = Field(default = None, description = "Suggestion 5 based upon the prompt")

parser = PydanticOutputParser(pydantic_object = suggestion)

state = pd.read_csv(r"testing.csv").loc[0, "State"]

try:
    vector_store_1 = FAISS.load_local(
        r"SIH_Vector_DB\FAISS_3_Pest_Weed_db", 
        embedding_model, 
        allow_dangerous_deserialization=True
    )
    
    print("All vector stores loaded successfully!")
    
except Exception as e:
    print(f"Error loading vector stores: {e}")

retriever_1 = vector_store_1.as_retriever(search_kwargs = {"k" : 1})

pest_data = retriever_1.invoke(state)[0].page_content

prompt = PromptTemplate(
    template = """You are a professional agricultural farmer, and you provided with a string of pests and weeds that were found in your fields. Your task is to find ways to eleminate these weeds and pests from you farms before they destroy the hard-earned yield. You are required to provide 5 suggestions based upon the details given upon how you will remove these pests and weeds from your farms. The suggestions should include names of the chemical or natural pasicides, insecticides etc. and other methods required to remove those unwanted things.
    You are not required to provide details from the string back to the farmer, and each suggestion should not be more than 3 lines. Keep in mind not to suggest too expensive needs that even the farmer fails to buy it.
    Data of the Pests and Weeds along with your location : {string}
    {format_instruction}""",
    input_variables = ["string"],
    partial_variables = {"format_instruction": parser.get_format_instructions()}
)

chain = prompt | model | parser

result = chain.invoke({"string" : pest_data})

print(result)