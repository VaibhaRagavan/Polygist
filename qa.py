import pc_store
import boto3
from langsmith import traceable

bedrock=boto3.client("bedrock-runtime",region_name="us-east-1")
@traceable
def answer(query,context):
    prompt=f"""You are answering a question using only the context provided below, which was retrieved from a document.
Context:
{context}
Question:
{query}
Instructions:
- Answer using only the information in the context above.
- If the context does not contain enough information to answer the question, say "I don't have enough information in the document to answer that" instead of guessing.
- Be concise and direct.
- Do not mention that you were given "context" or that this is a retrieval system — just answer naturally, as if you already know the document.
"""
    model_id="us.amazon.nova-2-lite-v1:0"
    message=[{
        "role":"user",
        "content":[{"text":prompt}]
    }]
    config={
        "maxTokens":2000,
        "temperature":0.3
    }
    try:
        response=bedrock.converse(
            modelId=model_id,
            messages=message,
            inferenceConfig=config,    )
        return response["output"]["message"]["content"][0]["text"]
    except Exception as e:
        return f"The Error:{e}"