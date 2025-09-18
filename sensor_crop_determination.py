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
    crop: str = Field(default = None, description = "Crop to be grown in the field")
    fact_1: str = Field(default = None, description = "Suggestion 1 based upon the prompt")
    fact_2: str = Field(default = None, description = "Suggestion 2 based upon the prompt")
    fact_3: str = Field(default = None, description = "Suggestion 3 based upon the prompt")
    fact_4: str = Field(default = None, description = "Suggestion 4 based upon the prompt")
    fact_5: str = Field(default = None, description = "Suggestion 5 based upon the prompt")

parser = PydanticOutputParser(pydantic_object = suggestion)

loader = CSVLoader(r"testing.csv", content_columns=["State","Nitrogen","Phosphorus","Potassium","Ph","Temperature","Salinity","Humidity"])
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
                if l[k] in crops:
                    continue
                else:
                    crops.append(l[k])
            break

retriever_3 = vector_store_3.as_retriever(search_kwargs = {"k" : 5})

results = retriever_3.invoke(testing[0].page_content)

crop_nutrients = ""
for i in results:
    crop_nutrients += i.page_content
print(crop_nutrients)

prompt = PromptTemplate(
    template = """You are a professional agricultural farmer, and you provided with a string of requirements present in the field. You are also provided with a context window of how much of each requirement is required by each crop below. Your task is to first decide the crop which will provide can grow on current conditions, give 5 suggestions upon what can be done in order to make the land more fertile for that particular crop, so that current conditions meet the ideal requirements.
    The suggestions should contain names of chemical fertilizers, manure, irrigation etc. or other methods to match the current condition with the ideal requirement. You are not required to provide details from the two strings back to the farmer, and each suggestion should not be more than 3 lines. Note that nutrients like Nitrogen, Phosphorus, Potassium are in kg/hectare, Temperature in Degrees Celsius, Salinity in dS/m, and Humidity in Percentages.
    String 1 (Data of the Current Field) : {string_1}
    String 2 (Ideal Requirement per crop) : {string_2}
    {format_instruction}""",
    input_variables = ["string_1", "string_2"],
    partial_variables = {"format_instruction": parser.get_format_instructions()}
)

chain = prompt | model | parser

result = chain.invoke({"string_1" : testing[0].page_content, "string_2" : crop_nutrients})

print(result)