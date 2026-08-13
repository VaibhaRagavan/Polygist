import streamlit as st
import extraction,pc_store,transcribe,qa,validation
import uuid
import os
from langsmith import traceable
from dotenv import load_dotenv
load_dotenv()
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY") or st.secrets["LANGSMITH_API_KEY"]
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "Polygist"
os.environ["LANGSMITH_ENDPOINT"] = "https://eu.api.smith.langchain.com"


if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

st.info(
    "  POLYGIST \n\n"
    "📄 Summary tab: PDF, image, audio, or video.\n\n "
    "🎙️ Audio/video limited to 30 minutes — longer files will be rejected.\n\n "
    "❓ Q&A tab: PDF only, one PDF per session — refresh the page to start a new PDF or session.\n\n"
    "Chat and summary results are not saved between refreshes."
)

tab1, tab2 = st.tabs(["Summary", "Q&A"])

@traceable
def pdf_summary(file):
    extract = extraction.pdf_extraction(file)
    summary = extraction.generate_summary(extract, "pdf")
    return summary

@traceable
def image_summary(file):
    details = extraction.image_extraction(file)
    return details

@traceable
def audio_video_summary(file, session_id):
    duration = validation.get_duration(file)
    if duration <= 1800:
        bucket_name = transcribe.bucket_name
        file_name = transcribe.upload_to_s3(file, session_id, bucket_name)
        job_name = transcribe.start_transcription_job(bucket_name, file_name)
        try:
            response = transcribe.get_transcription(job_name)
            summary = extraction.generate_summary(response, "video")
        except Exception as e:
            summary = f"Transcription failed:{e}"
        finally:
            transcribe.delete_from_s3(file_name, bucket_name)
        return summary
    else:
        return "Upload Video within 30mins duration"

with tab1:
    prompt = st.chat_input("Ask for the Summary", accept_file=True, file_type=["pdf", "jpeg", "png", "jpg", "mp3", "mp4", "mpeg"])

    if prompt and prompt.files:
        file = prompt.files[0]
        if file.type == "application/pdf":
            with st.spinner("Processing..."):
                st.session_state["summary_result"] = pdf_summary(file)

        elif file.type in ("image/jpeg", "image/png", "image/jpg"):
            with st.spinner("Processing..."):
                st.session_state["summary_result"] = image_summary(file)

        elif file.type in ("audio/mp3", "audio/mpeg", "video/mp4"):
            with st.spinner("Processing..."):
                st.session_state["summary_result"] = audio_video_summary(file, st.session_state.session_id)

        else:
            st.write("Unsupported file type")

    if st.session_state.get("summary_result"):
        st.write(st.session_state["summary_result"])


@traceable
def result_qa(query, file, session_id):
    if file and not st.session_state.get("qa_processed"):
        pdf_data = extraction.pdf_extraction(file)
        pdf_chunks = pc_store.Chunk_Text(pdf_data)
        pdf_embeb = [pc_store.Embedding(chunk) for chunk in pdf_chunks]
        store = pc_store.Vector_Store(pdf_chunks, pdf_embeb, session_id)
        st.session_state["qa_processed"] = True

    if query and st.session_state.get("qa_processed"):
        embed_query = pc_store.Embedding(query)
        retrived_data = pc_store.retrive(embed_query, session_id)
        result = qa.answer(query, retrived_data)
        return result
    return None

with tab2:
    prompt = st.chat_input("Enter your question and upload the pdf ", accept_file=True, file_type=["pdf"])
    if prompt:
        query = prompt.text
        files = prompt.files
        file = files[0] if files else None

        if (file and not st.session_state.get("qa_processed")) or (query and st.session_state.get("qa_processed")):
            with st.spinner("Processing..."):
                result = result_qa(query, file, st.session_state.session_id)
            if result:
                st.write(result)
        elif query and not st.session_state.get("qa_processed"):
            st.write("Please attach a pdf along with the your question first")