#Generate summary for the img,video,audio,pdf
import boto3
import pdfplumber
from langsmith import traceable

bedrock=boto3.client('bedrock-runtime', region_name='us-east-1')
#Image extraction 
@traceable

def image_extraction(image):
    image_bytes=image.read()
    image_format=image.type.split("/")[-1]
    model_id="us.amazon.nova-2-lite-v1:0"
    prompt=f"Desribe this image in 5line"
    message=[{
        "role":"user",
        "content":[{"text":prompt},
                   {"image":{
                       "format":image_format,
                       "source":{"bytes":image_bytes}
                   }}],
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
    
    
#Pdf extraction
@traceable
def pdf_extraction(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text_extracted=[]
        for pages in pdf.pages:
            text=pages.extract_text()
            if text:
                text_extracted.append(text)
        result="".join(text_extracted)
    return result

#Summary 
@traceable
def generate_summary(data:str,media_type:str):
    model_id="us.amazon.nova-2-lite-v1:0"
    if media_type == "video":
        prompt=f"""
        content:{data}
                Instructions:
                Detect whether this transcript is from a:
                   1) Meeting-Summarize this meeting transcript into concise bullet points highlighting action items, decisions, and next steps.
                   2) Hospital review-Summarize the patient’s feedback into a structured review highlighting positives, complaints, and suggestions.
                   3) Online class-Summarize this lecture into short notes, highlighting main concepts, examples, and key takeaways.
                   4) Other-summarize the content 
                 - Do not add extra information.
                Output format:
                    Only return the summary, do not include reasoning or instruction"""
    else:
        prompt=f"Generate the summary for the following text: {data} within 15 sentences"
     
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
