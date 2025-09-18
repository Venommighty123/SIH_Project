from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.document_loaders import CSVLoader
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
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
    fact_1: str = Field(default = None, description = "Suggestion 1 based upon the prompt"),
    fact_2: str = Field(default = None, description = "Suggestion 2 based upon the prompt"),
    fact_3: str = Field(default = None, description = "Suggestion 3 based upon the prompt"),
    fact_4: str = Field(default = None, description = "Suggestion 4 based upon the prompt"),
    fact_5: str = Field(default = None, description = "Suggestion 5 based upon the prompt"),

parser = PydanticOutputParser(pydantic_object = suggestion)

loader = CSVLoader(
    r"testing_2.csv",
    content_columns=["Crops","Nitrogen","Phosphorus","Potassium","Calcium","Magnesium","Sulfur","Iron","Manganese","Zinc","Copper","Boron","Molybdenum","Chlorine","Nickel","Silicon"])

testing_2 = loader.load()

try:
    vector_store_1 = FAISS.load_local(
        r"SIH_Vector_DB\FAISS_4_Crop_Nutrients_db", 
        embedding_model, 
        allow_dangerous_deserialization=True
    )
    
    print("Vector Store loaded successfully!")
    
except Exception as e:
    print(f"Error loading vector stores: {e}")

retriever_1 = vector_store_1.as_retriever(search_kwargs = {"k" : 1})

user_data = testing_2[0].page_content

result = retriever_1.invoke(user_data)

ideal_data = result[0].page_content

prompt = PromptTemplate(
    template = """You are an agricultural expert. You are provided with two strings: The first one contains the information about the soil field of the farmer, and the second string is the true information required to grow the particular crop.
    Your task is to give 5 suggestions the farmer should use in order to make his field better for the particular crop, based upon the two strings. The suggestions should contain names of chemical fertilizers, manure, irrigation etc. or other methods to match the nutrients with the ideal scenario. You are not required to provide details from the two strings back to the farmer, and each suggestion should not be more than 3 lines.
    String 1 :- {string_1}
    String 2 :- {string_2}
    {format_instruction}""",
    input_variables=["string_1", "string_2"],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

chain = prompt | model | parser

final_result = chain.invoke({"string_1" : user_data, "string_2" : ideal_data})

print(final_result)