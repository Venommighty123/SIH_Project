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
import random
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
    fact_1: str = Field(default = None, description = "Fact 1 based upon the prompt"),
    fact_2: str = Field(default = None, description = "Fact 2 based upon the prompt"),
    fact_3: str = Field(default = None, description = "Fact 3 based upon the prompt"),

structured_model = model.with_structured_output(suggestion)

loader = CSVLoader(r"testing.py")
testing = loader.load()
state = pd.read_csv(r"testing.csv").loc[0, "State"]

try:
    vector_store_1 = FAISS.load_local(
        r"SIH_Vector_DB\FAISS_1_State_Soil_db", 
        embedding_model, 
        allow_dangerous_deserialization=True
    )
    
    vector_store_2 = FAISS.load_local(
        r"SIH_Vector_DB\FAISS_2_Crop_Soil_db", 
        embedding_model, 
        allow_dangerous_deserialization=True
    )
    
    vector_store_3 = FAISS.load_local(
        r"SIH_Vector_DB\FAISS_3_Pest_Weed_db", 
        embedding_model, 
        allow_dangerous_deserialization=True
    )
    
    vector_store_4 = FAISS.load_local(
        r"SIH_Vector_DB\FAISS_4_Crop_Nutrients_db", 
        embedding_model, 
        allow_dangerous_deserialization=True
    )
    
    print("All vector stores loaded successfully!")
    
except Exception as e:
    print(f"Error loading vector stores: {e}")

retriever_1 = vector_store_1.as_retriever(search_kwargs = {"k" : 1})

soils_data = retriever_1.invoke(state)[0].page_content.split()[3:]

soils = ""

for i in soils_data:
    soils += i + " "

retriever_2 = vector_store_2.as_retriever(search_kwargs = {"k" : 3})

crop_data = retriever_2.invoke(soils)

crops = []

for i in crop_data:
    l = i.page_content.split()
    for j in range(len(l)):
        if l[j] == "Crops:":
            for k in range(j+1, len(l)):
                crops.append(l[k])
            break

crops = random.sample(crops, k = 3)

final_crops = ""

for i in crops:
    final_crops += i + " "

print(final_crops)

retriever_3 = vector_store_4.as_retriever(search_kwargs = {"k" : 3})

result = retriever_3.invoke(final_crops)

print(result)