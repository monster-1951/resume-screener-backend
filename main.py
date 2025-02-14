# 3
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
# 2
import spacy
import subprocess
import re
# 1
from fastapi import FastAPI, UploadFile, File, Form
import fitz  # PyMuPDF
from pdfminer.high_level import extract_text
from docx import Document
from typing import Optional
from io import BytesIO  # Import BytesIO to handle byte streams

app = FastAPI()


# Load SpaCy NLP model (Replace with your model if different)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")
# Load BERT model (Sentence-BERT)
bert_model = SentenceTransformer("all-MiniLM-L6-v2")

def compute_similarity_bert(resume_text, job_text, resume_skills, job_skills):
    if not job_text:
        return {"error": "No job description provided"}

    # Encode Resume & Job Description using BERT
    resume_embedding = bert_model.encode(resume_text, convert_to_tensor=True)
    job_embedding = bert_model.encode(job_text, convert_to_tensor=True)
    text_similarity = cosine_similarity([resume_embedding.cpu().numpy()], [job_embedding.cpu().numpy()])[0][0]

    # Skill Matching (Jaccard Similarity)
    resume_skills_set = set(resume_skills)
    job_skills_set = set(job_skills)
    skill_match_score = len(resume_skills_set & job_skills_set) / len(resume_skills_set | job_skills_set) if job_skills_set else 0

    # Weighted Score: 70% BERT Similarity, 30% Skill Match
    final_score = (0.7 * text_similarity) + (0.3 * skill_match_score)

    return round(final_score * 100, 2)

job_titles = ["Software Engineer", "Python Developer", "Data Scientist", "Machine Learning Engineer", 
              "AI Engineer", "Backend Developer", "NLP Engineer"]

def extract_entities(text):
    doc = nlp(text)
    extracted_info = {
        "Name": None,
        "Email": None,
        "Phone": None,
        "Education": [],
        "Experience": [],
        "Skills": []
    }

    # Extract Email
    email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    if email_match:
        extracted_info["Email"] = email_match.group()

    # Extract Phone Number
    phone_match = re.search(r"\+?\d[\d\s\-\(\)]{8,15}\d", text)
    if phone_match:
        extracted_info["Phone"] = phone_match.group()

    # Extract Name (First "PERSON" entity)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            extracted_info["Name"] = ent.text.strip()
            break

    # Extract Education (Find degrees/universities)
    education_keywords = ["Bachelor", "Master", "PhD", "University", "College", "Institute"]
    for ent in doc.ents:
        if any(keyword in ent.text for keyword in education_keywords):
            extracted_info["Education"].append(ent.text)

    # Extract Experience (Job Titles + Company)
    experience = []
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if any(title in line for title in job_titles):
            company_line = lines[i + 1] if i + 1 < len(lines) else ""
            experience.append(f"{line} at {company_line.strip()}")
    extracted_info["Experience"] = experience

    # Extract Skills (Match from predefined list)
    skills_list = ["Python", "JavaScript", "SQL", "FastAPI", "Flask", "Django", "NLP", 
                   "SpaCy", "BERT", "GPT-4", "PostgreSQL", "MongoDB", "Docker", "Git", "AWS"]
    extracted_info["Skills"] = [skill for skill in skills_list if skill.lower() in text.lower()]

    return extracted_info
# Function to extract text from PDF (PyMuPDF)
def extract_text_from_pdf(file_bytes):
    file_stream = BytesIO(file_bytes)  # Convert bytes to a file-like object
    doc = fitz.open(stream=file_stream, filetype="pdf")
    text = "\n".join([page.get_text("text") for page in doc])
    return text

# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text


@app.get("/")
def read_root():
    return {"message": "AI Resume Screener API is running! on port "}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
# API to upload resume & job description
@app.post("/upload-resume/")
async def upload_resume(
    resume: UploadFile = File(...),
    job_description: Optional[str] = Form(None),
    job_file: Optional[UploadFile] = File(None)
):
    # Read the uploaded resume file as bytes
    resume_bytes = await resume.read()

    # Extract text from the resume (PDF or DOCX)
    if resume.filename.endswith(".pdf"):
        resume_text = extract_text_from_pdf(resume_bytes)  # Pass bytes
    elif resume.filename.endswith(".docx"):
        resume_text = extract_text_from_docx(BytesIO(resume_bytes))  # Convert bytes to file-like object
    else:
        return {"error": "Unsupported file format. Upload PDF or DOCX."}

    # Extract job description (from form input or file)
    job_text = job_description
    if job_file:
        job_bytes = await job_file.read()
        if job_file.filename.endswith(".pdf"):
            job_text = extract_text_from_pdf(job_bytes)
        elif job_file.filename.endswith(".docx"):
            job_text = extract_text_from_docx(BytesIO(job_bytes))
        else:
            return {"error": "Unsupported job description format. Use PDF/DOCX."}

    # Extract entities using NLP
    parsed_data = extract_entities(resume_text)

     # Compute similarity score
    job_skills = ["Python", "FastAPI", "Flask", "BERT", "NLP", "PostgreSQL", "MongoDB"]  # Extract from job_text later
    match_score = compute_similarity_bert(resume_text, job_text, parsed_data["Skills"], job_skills) if job_text else None

     # Convert match_score from numpy.float32 to float for JSON serializability
    match_score = float(match_score) if match_score is not None else None
    

    print(match_score)
    return {
        "resume_text": resume_text[:500],  # Limit preview to 500 chars
        "job_description": job_text[:500] if job_text else "No job description provided",
        "parsed_resume":parsed_data,
        "match_score": match_score,
    }
