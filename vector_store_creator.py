from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import os

load_dotenv()

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

loader_1 = CSVLoader(r"Crop Files\State_Soil.csv")
state_soil = loader_1.load()

loader_2 = CSVLoader(r"Crop Files\Crop_Soil.csv")
crop_soil = loader_2.load()

loader_3 = CSVLoader(r"Crop Files\State_Pest_Weed.csv")
pest_weed = loader_3.load()

loader_4 = CSVLoader(r"Crop Files\New_Crop_Nutrients.csv")
crop_nutrients = loader_4.load()

os.makedirs(r"SIH_Vector_DB\FAISS_1_State_Soil_db", exist_ok=True)
os.makedirs(r"SIH_Vector_DB\FAISS_2_Crop_Soil_db", exist_ok=True)
os.makedirs(r"SIH_Vector_DB\FAISS_3_Pest_Weed_db", exist_ok=True)
os.makedirs(r"SIH_Vector_DB\FAISS_4_Crop_Nutrients_db", exist_ok=True)

vector_store_1 = FAISS.from_documents(
    documents=state_soil,
    embedding=embedding_model
)
vector_store_1.save_local(r"SIH_Vector_DB\FAISS_1_State_Soil_db")

vector_store_2 = FAISS.from_documents(
    documents=crop_soil,
    embedding=embedding_model
)
vector_store_2.save_local(r"SIH_Vector_DB\FAISS_2_Crop_Soil_db")

vector_store_3 = FAISS.from_documents(
    documents=pest_weed,
    embedding=embedding_model
)
vector_store_3.save_local(r"SIH_Vector_DB\FAISS_3_Pest_Weed_db")


vector_store_4 = FAISS.from_documents(
    documents=crop_nutrients,
    embedding=embedding_model
)
vector_store_4.save_local(r"SIH_Vector_DB\FAISS_4_Crop_Nutrients_db")
