import boto3
import os, requests
import time
from langsmith import traceable
from dotenv import load_dotenv
load_dotenv()

s3=boto3.client("s3",region_name="eu-west-1")
transcribe=boto3.client("transcribe",region_name="eu-west-1")

bucket_name=os.getenv("S3_BUCKET_NAME")
@traceable
def upload_to_s3(file_path,ext,sessionid,bucket_name):
    file_name=f"{sessionid}.{ext}"
    with open(file_path,"rb") as f:
        file_bytes=f.read()
        s3.put_object(
            Bucket=bucket_name,
            Key=file_name,
            Body=file_bytes
    )
    print(f"Successfully created {file_name} in {bucket_name}")
    return file_name

##Transcription 
@traceable
def start_transcription_job(bucket,key):
    job_name=f"transcription-job-{int(time.time())}"
    media=f"s3://{bucket}/{key}"

    response=transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={'MediaFileUri':media},
        MediaFormat=key.split(".")[-1],
        LanguageCode="en-US"    
    )
    print(response["TranscriptionJob"]["TranscriptionJobName"])
    return response["TranscriptionJob"]["TranscriptionJobName"]

##Get the Transcription
@traceable
def get_transcription(job_name,max_wait=300,poll_interval=5):
    waited=0
    while waited<max_wait:
        result=transcribe.get_transcription_job(TranscriptionJobName=job_name)
        status=result["TranscriptionJob"]["TranscriptionJobStatus"]
        if status=="COMPLETED":
            uri=result["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
            r=requests.get(uri)
            transcription=r.json()
            transcript_data=transcription["results"]["transcripts"][0]["transcript"]
            return transcript_data
        elif status=="FAILED":
            raise Exception("Transcription job failed")
        time.sleep(poll_interval)
        waited+=poll_interval
    raise TimeoutError("Transcription job did not completed on time")   

##Delete media from s3
@traceable
def delete_from_s3(file_name,bucket_name):
    s3.delete_object(
        Bucket=bucket_name,
        Key=file_name,
    )
    print("The media removed from the bucket")
    return
