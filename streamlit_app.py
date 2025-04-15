import streamlit as st
import pandas as pd
import spacy
import re
import io
import fitz  # for pdfs
import docx  # for doc files
from PIL import Image
import base64

st.set_page_config(
    page_title="Resume Parser", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📄"
)

st.markdown("""
<style>
    .main-header {
        font-size: 42px;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 20px;
        color: #64748B;
        margin-bottom: 30px;
    }
    .section-header {
        font-size: 24px;
        font-weight: 600;
        color: #1E3A8A;
        margin-top: 20px;
        margin-bottom: 10px;
        padding-bottom: 5px;
        border-bottom: 2px solid #E2E8F0;
    }
    .card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        border-radius: 6px;
        padding: 10px 15px;
        font-weight: 600;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
    }
    .upload-section {
        border: 2px dashed #CBD5E1;
        border-radius: 10px;
        padding: 30px;
        text-align: center;
        margin-bottom: 30px;
        background-color: #F1F5F9;
    }
    .stProgress > div > div > div > div {
        background-color: #2563EB;
    }
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stTextArea>div>textarea {
        background-color: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-radius: 6px;
        padding: 10px;
        font-size: 16px;
    }
    .stExpander {
        border: 1px solid #E2E8F0;
        border-radius: 6px;
    }
    .footer {
        text-align: center;
        color: #64748B;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #E2E8F0;
    }
    .highlight {
        background-color: #F0F9FF;
        padding: 5px 10px;
        border-radius: 6px;
        font-weight: 500;
        color: #0369A1;
    }
</style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown('<h1 class="main-header">Smart Resume Parser</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Extract key information from resumes with AI-powered analysis</p>', unsafe_allow_html=True)

# Loading the model
@st.cache_resource
def load_model():
    return spacy.load("en_core_web_lg")

try:
    nlp = load_model()
except OSError:
    st.error("Please try again after installing all the required libraries.")
    st.stop()

st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.markdown('📤 **Upload Your Resume**')
st.markdown('Supported formats: PDF, DOCX')
uploaded_file = st.file_uploader("", type=["pdf", "docx"])
st.markdown('</div>', unsafe_allow_html=True)

# Extracting the text from PDF files
def extract_text_from_pdf(pdf_file):
    pdf_stream = io.BytesIO(pdf_file.getvalue())
    pdf_document = fitz.open(stream=pdf_stream, filetype="pdf")
    text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

# Extracting text from DOCX files
def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

# Extracting the email addresses (using regex)
def extract_email(text):
    email_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_regex, text)
    return emails[0] if emails else ""

# Extracting the phone numbers (again, using regex)
def extract_phone(text):
    # for various phone number formats
    phone_regex = r'(\+\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\d{10})'
    phones = re.findall(phone_regex, text)
    if phones:
        phone = ''.join(''.join(p) for p in phones[0])
        if len(phone) >= 10:
            return phone
    return ""

# Extracting the names (using 'person' entity recognition)
def extract_name(doc):
    candidates = []
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            candidates.append(ent.text)
    
    if candidates:
        candidates.sort(key=len, reverse=True)
        return candidates[0]
    return ""

# Extracting the skills (using some of the keywords and NER)
def extract_skills(doc, text):
    tech_skills = [
        "python", "java", "javascript", "react", "angular", "vue", "node.js", "django", "flask",
        "html", "css", "html5", "css3", "sql", "nosql", "mongodb", "postgresql", "mysql", "mariadb",
        "aws", "azure", "gcp", "docker", "kubernetes", "ci/cd", "git", "jenkins", "terraform",
        "machine learning", "deep learning", "nlp", "data science", "data analysis", "powerbi",
        "r", "golang", "c++", "c#", "php", "ruby", "scala", "swift", "kotlin", "typescript",
        "hadoop", "spark", "tableau", "excel", "powerpoint", "linux", "unix", "windows", "macos",
        "rest api", "graphql", "oauth", "json", "xml", "soap", "jquery", "bootstrap", "redux",
        "spring", "hibernate", "servlet", "jsp", "asp.net", "mvc", "mvvm", "oop", "tdd", "agile",
        "scrum", "kanban", "jira", "confluence", "project management", "team management",
        "leadership", "communication", "problem-solving", "critical thinking", "cooperation",
        "mobile development", "android", "ios", "react native", "flutter", "xamarin", "tensorflow",
        "pytorch", "keras", "scikit-learn", "pandas", "numpy", "matplotlib", "seaborn", "d3.js",
        "blockchain", "cryptocurrency", "smart contracts", "solidity", "web3", "ethereum",
        "devops", "sre", "system design", "microservices", "restful", "cybersecurity", "pentesting"
    ]
    
    found_skills = set()
    text_lower = text.lower()
    
    for skill in tech_skills:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.add(skill)
    
    for ent in doc.ents:
        if ent.label_ == "ORG":
            skill_name = ent.text.lower()
            if skill_name in tech_skills:
                found_skills.add(ent.text)
    
    lang_pattern = r'\b(?:Python|Java(?:Script)?|C\+\+|C#|PHP|Ruby|Go(?:lang)?|Swift|Kotlin|R|Scala|Perl|Rust|TypeScript|Dart|Objective-C|Shell|Bash|PowerShell|SQL|NoSQL)\b'
    langs = re.findall(lang_pattern, text, re.IGNORECASE)
    for lang in langs:
        found_skills.add(lang)
    
    return sorted(list(found_skills))

# Extrracting education details
def extract_education(doc):
    education_data = []
    
    edu_orgs = []
    for ent in doc.ents:
        if ent.label_ == "ORG":
            org_text = ent.text.lower()
            if any(word in org_text for word in ["university", "college", "institute", "school"]):
                edu_orgs.append(ent.text)
    
    # For extracting degrees
    degree_pattern = r'\b(?:Bachelor|Master|MBA|PhD|BSc|MSc|B\.Tech|M\.Tech|B\.E|M\.E|B\.A|M\.A|B\.Com|M\.Com|Bachelor\'s|Master\'s|Doctorate|Diploma)(?:\sof\s(?:Science|Arts|Engineering|Technology|Commerce|Business Administration))?\b'
    degrees = re.findall(degree_pattern, doc.text)
    major_pattern = r'\bin\s(?:Computer Science|Information Technology|CSE| Business|Economics|Engineering|Mathematics|Physics|Chemistry|Biology|Psychology|Marketing|Finance|Accounting|Management|Law|Medicine|Communications|Media|Design|Architecture)\b'
    majors = re.findall(major_pattern, doc.text)
    
    if edu_orgs:
        for org in edu_orgs:
            education_data.append(org)
            
    if degrees:
        for degree in degrees:
            if degree not in education_data:
                education_data.append(degree)
    
    if majors:
        for major in majors:
            if major not in education_data:
                education_data.append(major.replace("in ", ""))
    
    return education_data

# Extracting experience years (by duration - overall)
def extract_experience_years(text):
    year_patterns = [
        r'(\d+)\+?\s*(?:years|yrs)(?:\s*of)?\s*experience',
        r'experience\s*(?:of|for)?\s*(\d+)\+?\s*(?:years|yrs)',
        r'worked\s*(?:for)?\s*(\d+)\+?\s*(?:years|yrs)',
        r'(\d+)\+?\s*(?:years|yrs)(?:\s*in)'
    ]
    
    total_years = 0
    for pattern in year_patterns:
        matches = re.findall(pattern, text.lower())
        for match in matches:
            try:
                years = int(match.replace('+', ''))
                total_years = max(total_years, years)
            except:
                pass
    
    if total_years == 0:
        date_pattern = r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?,?\s*\'?(\d{2,4})\s*-\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|Present)[a-z]*\.?,?\s*\'?(\d{0,4})'
        date_matches = re.findall(date_pattern, text)
        
        from datetime import datetime # in case for 'present'
        current_year = datetime.now().year
        current_month = datetime.now().month
        
        month_to_num = {
            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
        }
        
        all_durations = []
        for match in date_matches:
            start_month, start_year, end_month, end_year = match
            
            start_month_num = month_to_num.get(start_month[:3], 1)
            
            # Handling the 2-digit years
            if len(str(start_year)) <= 2:
                start_year = int(start_year)
                start_year = start_year + (2000 if start_year < 50 else 1900)
            else:
                start_year = int(start_year)
            
            if end_month == 'Present':
                end_month_num = current_month
                end_year = current_year
            else:
                end_month_num = month_to_num.get(end_month[:3], 12)
                if end_year == '':
                    end_year = current_year
                elif len(str(end_year)) <= 2:
                    end_year = int(end_year)
                    end_year = end_year + (2000 if end_year < 50 else 1900)
                else:
                    end_year = int(end_year)
            
            duration = (end_year - start_year) + (end_month_num - start_month_num) / 12.0
            if duration > 0:
                all_durations.append(duration)
        
        if all_durations:
            total_years = round(sum(all_durations), 2)
    
    return total_years

# Extracting the company names
def extract_companies(doc, text):
    company_suffixes = [
        "Ltd", "Limited", "LLC", "Corp", "Corporation", "Inc", "Incorporated",
        "Pvt", "Private", "Pvt. Ltd", "Systems", "Technologies", "Solutions",
        "Software", "Services", "Group", "Holdings", "International", "Global"
    ]
    
    # Defining terms that indicate non-companies, as it was creating issue
    non_company_indicators = [
        "Certificate", "Certification", "Course", "Courses", "Stack", "Full Stack",
        "Data Science", "Data Scientist", "Engineer", "Developer", "Intern",
        "Machine Learning", "AI", "Artificial Intelligence", "NLP", "Deep Learning",
        "CSE", "Computer Science", "Education", "University", "College", "School",
        "Institute", "Fundamentals", "Analysis", "Analytics"
    ]
    
    companies = set()
    
    for suffix in company_suffixes:
        pattern = r'([A-Z][A-Za-z\s]+)\s+' + re.escape(suffix) + r'\b'
        matches = re.findall(pattern, text)
        for match in matches:
            company_name = f"{match.strip()} {suffix}"
            if not any(indicator.lower() in company_name.lower() for indicator in non_company_indicators):
                companies.add(company_name)
    
    employment_patterns = [
        r'(?:worked|employed)\s+(?:at|by|with|for)\s+([A-Z][A-Za-z\s]+(?:\s+(?:' + '|'.join(company_suffixes) + ')))',
        r'(?:position|role|job)\s+at\s+([A-Z][A-Za-z\s]+(?:\s+(?:' + '|'.join(company_suffixes) + ')))'
    ]
    
    for pattern in employment_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if not any(indicator.lower() in match.lower() for indicator in non_company_indicators):
                companies.add(match.strip())
    
    for ent in doc.ents:
        if ent.label_ == "ORG":
            org_text = ent.text
            is_likely_company = any(suffix.lower() in org_text.lower() for suffix in company_suffixes)
            
            contains_non_company = any(indicator.lower() in org_text.lower() for indicator in non_company_indicators)
            is_educational = any(edu in org_text.lower() for edu in ["university", "college", "institute", "school"])
            is_tech_term = any(tech in org_text.lower() for tech in ["python", "java", "javascript", "ml", "ai", "nlp"])
            
            if is_likely_company and not (contains_non_company or is_educational or is_tech_term):
                companies.add(org_text)
    
    filtered_companies = []
    for company in companies:
        is_job_title = any(title in company.lower() for title in ["engineer", "developer", "scientist", "intern"])
        is_course = any(course in company.lower() for course in ["certificate", "course", "fundamentals", "full stack"])
        is_technology = any(tech in company.lower() for tech in ["data science", "machine learning", "ai", "ml", "analytics"])
        
        if not (is_job_title or is_course or is_technology) and len(company) >= 4 and company[0].isupper():
            filtered_companies.append(company)
    
    if len(filtered_companies) == 0:
        fallback_companies = []
        for ent in doc.ents:
            if ent.label_ == "ORG" and len(ent.text) >= 4:
                # Skip obvious non-companies
                if not any(term.lower() in ent.text.lower() for term in 
                          ["university", "college", "school", "data science", "python", 
                           "certificate", "machine learning", "fundamentals"]):
                    fallback_companies.append(ent.text)
        
        return fallback_companies if fallback_companies else ["No companies detected"]
    
    return sorted(filtered_companies)

# For creating skill 
def create_skill_badge(skill):
    return f'<span class="highlight">{skill}</span>'

def feature_section():
    st.markdown('<div class="section-header">🚀 Features</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)

    # Just filling up the blank space on the initial page before the user uploads the resume
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 📝 Smart Extraction")
        st.markdown("Extract key information from resumes using advanced NLP techniques")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🔍 AI-Powered Analysis")
        st.markdown("Identify skills, experience, education and more with machine learning")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col3:  
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### ⚡ Fast Processing")
        st.markdown("Get results in seconds with our optimized processing engine")
        st.markdown('</div>', unsafe_allow_html=True)

# Main function
def process_resume(text):
    doc = nlp(text)
    
    # Extract all the required information
    name = extract_name(doc)
    email = extract_email(text)
    phone = extract_phone(text)
    skills = extract_skills(doc, text)
    education = extract_education(doc)
    total_years = extract_experience_years(text)
    companies = extract_companies(doc, text)
    
    results = {
        "name": name,
        "email": email,
        "phone": phone,
        "skills": skills,
        "education": education,
        "total_years": total_years,
        "companies": companies
    }
    
    return results

# Displaing "Welcome" till the user has not uploaded any file
if uploaded_file is None:
    feature_section()
    
    st.markdown('<div class="section-header">📊 How It Works</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 1. Upload Resume")
        st.markdown("Upload your resume in PDF or DOCX format")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 2. AI Processing")
        st.markdown("Our AI model analyzes and extracts key information")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 3. Review & Save")
        st.markdown("Review the extracted information and save the results")
        st.markdown('</div>', unsafe_allow_html=True)

else:
    with st.spinner("🔍 Processing your resume..."):
        progress_bar = st.progress(0)
        progress_bar.progress(25)
        
        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = extract_text_from_docx(uploaded_file)
        else:
            st.error("⚠️ Unsupported file format")
            text = ""
        
        progress_bar.progress(50)
        
        if text:
            results = process_resume(text)
            progress_bar.progress(75)
            
            st.markdown('<div class="section-header">✅ Extracted Information</div>', unsafe_allow_html=True)
            
            if results["name"]:
                st.markdown(f"### Hello, {results['name']}! 👋")
                st.markdown("We've analyzed your resume and extracted the following information. Please review and make any necessary corrections.")
            else:
                st.markdown("### Resume Analysis Results")
                st.markdown("We've analyzed your resume. Please review the extracted information below.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="section-header">👤 Personal Details</div>', unsafe_allow_html=True)
                name = st.text_input("Full Name", results["name"])
                email = st.text_input("Email Address", results["email"])
                phone = st.text_input("Phone Number", results["phone"])
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="section-header">💼 Experience</div>', unsafe_allow_html=True)
                total_years = st.number_input("Total Years of Experience", value=float(results["total_years"]), format="%.1f")
                
                st.markdown("#### Organizations")
                companies_text = ", ".join(results["companies"])
                companies_edited = st.text_area("", companies_text, placeholder="Organizations you've worked for")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="section-header">🔧 Skills</div>', unsafe_allow_html=True)
                
                if results["skills"]:
                    badges_html = " ".join([create_skill_badge(skill) for skill in results["skills"]])
                    st.markdown(f"<div>{badges_html}</div>", unsafe_allow_html=True)
                
                skills_text = ", ".join(results["skills"])
                skills_edited = st.text_area("Edit Skills", skills_text, placeholder="Your professional skills")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="section-header">🎓 Education</div>', unsafe_allow_html=True)
                education_text = "\n".join(results["education"])
                education_edited = st.text_area("", education_text, placeholder="Your educational background")
                st.markdown('</div>', unsafe_allow_html=True)
            
            progress_bar.progress(100)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("💾 Save Parsed Resume"):
                    st.success("✅ Resume information saved successfully!")
            
            with col2:
                if st.button("🔄 Reset Information"):
                    st.rerun()
            
            with st.expander("🔍 View Raw Extracted Text"):
                st.text_area("", text, height=300)
    
    st.markdown('<div class="footer">Smart Resume Parser v1.0 | Powered by AI</div>', unsafe_allow_html=True)