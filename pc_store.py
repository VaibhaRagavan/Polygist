import json
import boto3,os
from dotenv import load_dotenv
load_dotenv()
from pinecone import Pinecone
from datetime import datetime,timezone
from langsmith import traceable
import streamlit as st

bedrock=boto3.client("bedrock-runtime",region_name="us-east-1")

key=os.getenv("PINECONE_API_KEY") or st.secrets["PINECONE_API_KEY"]
pc=Pinecone(api_key= key)
Index_Name="polygist"
index=pc.Index(Index_Name)

#Chunking
@traceable
def Chunk_Text(text, chunk_size=2000, overlap=200):

    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start : (start+chunk_size)]  
        chunks.append(chunk)
        start = start+(chunk_size - overlap)               
    return chunks

#Embedding

def Embedding(text):
    request_body={
        "inputText":text,
        "dimensions":1024,
        "normalize": True
    }
    body_string=json.dumps(request_body)
    response=bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=body_string
    )
    response_body=json.loads(response["body"].read())
    return response_body["embedding"]

#Pinecone store
@traceable
def Vector_Store(chunks,embeds,session_id):
        try:
            if not embeds or not chunks:
                raise ValueError("vectors and chunks required")
            record=[]
            for i,(embed,chunk) in enumerate(zip(embeds,chunks)):
    
                record.append({
                    "id":f"chunk-{i}",
                    "values":embed,
                    "metadata":{
                        "text":chunk,
                        "created_at":datetime.now(timezone.utc).timestamp()
                    }
                })
            start=0
            while start < len(record):
                ind_sert=record[start:(start+500)]
                index.upsert(vectors=ind_sert,namespace=session_id,)
                start += 500
            print("Data stored as vector")
            return f"Data stored as vector"
        except Exception as e:
            print("Error",e)

##Retrival
@traceable
def retrive(query_embed,sessionid):
    result=index.query(
          vector=query_embed,
          top_k=3,
          namespace=sessionid,
          include_values=False,
          include_metadata=True,
          )
    resp=[]
    for match in result["matches"]:
        text=match["metadata"].get("text","No text provided")
        resp.append(text)
    return resp


