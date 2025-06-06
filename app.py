#!/usr/bin/env python
import cv2
import json
import numpy as np
import face_classifier
import tensorflow as tf

from flask import Flask, render_template,  request, send_from_directory, redirect, url_for
from keras.models import model_from_json, load_model
from flask import Flask, jsonify, request, render_template
import pandas as pd
import utility_functions
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import face_functions

from werkzeug.utils import secure_filename
import helper
from sentence_transformers import SentenceTransformer, util

from flask import request
import json
import spacy

from googlesearch import search
from bs4 import BeautifulSoup
import requests
import nltk
  
#GEMINI code
import re
import os
import google.generativeai as genai
from dotenv import load_dotenv
# Load environment variables from .env
load_dotenv()

safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]
GOOGLE_API_KEY = os.getenv('GEMINI_API_KEY')  # Replace with your actual API key
genai.configure(api_key=GOOGLE_API_KEY)
gem_model = genai.GenerativeModel('gemini-1.5-flash',safety_settings=safety_settings)


# df = pd.read_csv("Question Dataset.csv")
df = pd.read_csv("questions_csv.csv")
pd.set_option('display.max_colwidth', None)

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Load Haarcascade File
face_detector = cv2.CascadeClassifier("ml_folder/haarcascade_frontalface_default.xml")

# Load the Model and Weights
model = load_model('ml_folder/video.h5')
# model.load_weights('ml_folder/model.h5')
# model._make_predict_function()

skills_from_resume=['object oriented programming','stacks','database management','sql','computer networks','operating systems','queues',
                    'trees','arrays','programming systems']
report_skills = []
final_skills = []

nlp = spacy.load('model-best')

def parse_resume(file):
    text = helper.extract_text_from_pdf(file)
    # print("text " ,text)
    doc = nlp(text)
    entities = [{'text': ent.text, 'label': ent.label_} for ent in doc.ents]
    for ent in doc.ents:
        if(ent.label_ == 'SKILLS'):
            skills_from_resume.append(ent.text)
    print(skills_from_resume)
    return entities

nltk.download('punkt')

def scrape_questions_from_search(query, num_results=5, max_questions_per_site=2):
    search_results = search(query, num_results=num_results)

    questions = []

    for url in search_results:
        try:
            # Send a GET request to the URL with a User-Agent header
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers)

            # Check if the request was successful (status code 200)
            if response.status_code == 200:
                # Parse the HTML content of the page
                soup = BeautifulSoup(response.text, 'html.parser')

                # Extract all text elements from the HTML document
                all_text = soup.get_text(separator=' ')

                # Tokenize the text into sentences and filter those starting with common question words and ending with a question mark
                sentences = nltk.sent_tokenize(all_text)
                
                # Extract up to max_questions_per_site questions from each website
                extracted_questions = [sentence.strip() for sentence in sentences if sentence.strip().lower().startswith(('what','state','in','define','explain', 'why', 'how', 'when', 'where', 'which', 'is', 'are', 'do', 'does', 'did')) and sentence.strip().endswith('?')][:max_questions_per_site]
                
                questions.extend(extracted_questions)

                # If the total number of questions reaches the desired limit, break out of the loop
                if len(questions) >= num_results:
                    break

        except Exception as e:
            print(f"Error processing URL {url}: {e}")

    return questions


def context(Skills, final_exp):
    if Skills not in report_skills:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        sentences1 = [final_exp]
        sentences2 = [Skills]

        embeddings1 = model.encode(sentences1, convert_to_tensor=True)
        embeddings2 = model.encode(sentences2, convert_to_tensor=True)
        cosine_scores = util.cos_sim(embeddings1, embeddings2)
        if cosine_scores[0][0] > 0.45:
            report_skills.append(Skills)
            print(Skills, final_exp, cosine_scores[0][0])


def context_job(Skills, final_exp):
    if Skills not in final_skills:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        sentences1 = [final_exp]
        sentences2 = [Skills]

        embeddings1 = model.encode(sentences1, convert_to_tensor=True)
        embeddings2 = model.encode(sentences2, convert_to_tensor=True)
        cosine_scores = util.cos_sim(embeddings1, embeddings2)
        if cosine_scores[0][0] > 0.45:
            final_skills.append(Skills)
            print(Skills, final_exp, cosine_scores[0][0])



def start_file(job_description, filename):
    if filename[len(filename)-1] == 'x':
        text = helper.extract_text_from_docx(filename)
    else:
        text = helper.extract_text_from_pdf(filename)

    cleaned_text = helper.clean_text(text)

    str_resume = ""
    for i in cleaned_text:
        str_resume += i
        str_resume += " "
    # print(str_resume)

    doc = helper.get_nlp(str_resume)
    # print(doc)

    # get profile section
    sec_profile = []
    sec_profile = helper.get_profile_section(doc)
    # print(sec_profile)

    # get education section
    sec_education = []
    sec_education = helper.get_education_section(doc)
    # print(sec_education)

    # get experience section
    sec_experience = []
    sec_experience = helper.get_experience_section(doc)
    # print(sec_experience)

    # get skills section
    sec_skills = []
    sec_skills = helper.get_skills_section(doc)
    # print(sec_skills)

    # get extra section
    sec_extra = []
    sec_extra = helper.get_extra(doc)
    # print(sec_extra)

    for i in sec_skills:
        for j in sec_experience:
            context(i, j)

    print("Report Skills: \n", report_skills)
    print()

    # Job description part
    text_list = helper.extract_text_from_docx(job_description)
    text_list = text_list.replace('\n', ' ')

    job_doc = helper.get_nlp(text_list)
    # print(job_doc)
    req_tokens = helper.get_requirements(job_doc)
    # print(req_tokens)
    # print()

    for i in report_skills:
        for j in req_tokens:
            context_job(i, j)


    print("Final Skills: ", final_skills)

    experience_str = ""
    for i in sec_experience:
        experience_str += i
        experience_str += ", "
    exp_summary = helper.get_summary(experience_str)

    report_exp = ""
    for i in exp_summary:
        report_exp += i
        report_exp += ", "


    skill_per = (len(final_skills) / len(req_tokens))*100
    print("Skills Percentage: ", skill_per)


    # Report Section


    profile_str = ""
    for i in sec_profile:
        profile_str += i
        profile_str += " "

    education_str = ""
    for i in sec_education:
        education_str += i
        education_str += " "

    experience_str2 = ""
    for i in exp_summary:
        experience_str2 += i
        experience_str2 += " "

    skills_str = ""
    for i in final_skills:
        skills_str += i
        skills_str += " "

    extra_str = ""
    for i in sec_extra:
        extra_str += i
        extra_str += " "



    report_file = open("report_file.txt", "w", encoding="utf-8")
    report_file.write("Profile: \n")
    report_file.write(profile_str)
    report_file.write("\n")
    report_file.write("\n")
    report_file.write("Education: \n")
    report_file.write(education_str)
    report_file.write("\n")
    report_file.write("\n")
    report_file.write("Experience: \n")
    report_file.write(report_exp)
    report_file.write("\n")
    report_file.write("\n")
    report_file.write("Skills: \n")
    report_file.write(skills_str)
    report_file.write("\n")
    report_file.write("\n")
    report_file.write("Extra: \n")
    report_file.write(extra_str)
    report_file.write("\n")
    report_file.write("\n")
    report_file.write("Resume matched ")
    report_file.write(str(skill_per))
    report_file.write(" % with job description")
    report_file.write("\n")

    if skill_per > 40:
        report_file.write("Candidate Qualified for Interview Round\n")
        report_file.write("\n")
        report_file.write("\n")


def parseJobDescription(filename):
    if filename[len(filename)-1] == 'x':
        text = helper.extract_text_from_docx(filename)
    else:
        text = helper.extract_text_from_pdf(filename)

   
    #  # Job description part
    # text_list = helper.extract_text_from_docx(job_description)
    # text_list = text_list.replace('\n', ' ')

    job_doc = helper.get_nlp(text)
    # print(job_doc)
    req_tokens = helper.get_requirements(job_doc)
    print(req_tokens)
    return req_tokens
    # print()
    
    
def extract_questions(text):
    
  # Improved regular expression for better question capture
  question_pattern = r"\d+\. (.*?)(?:\n|\Z)"  # Matches questions starting with a number and ending with newline or end of text

  all_questions = []
  sections = text.split("\n\n")  # Split by empty lines to separate sections

  for section in sections:
    if section.strip():  # Check if section is not empty
      topic = section.split(":")[0].strip()  # Extract topic (assuming format "Topic:")
      matches = re.findall(question_pattern, section)
      for match in matches:
        question = match.strip()
        all_questions.append(question)  # Store topic and question together

  return all_questions

def answer_query(query):
    try:
        # Configure the model with your API key (replace with your actual key)
        response = gem_model.generate_content(query)
        return response.text
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Failed to generate response."
    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/resume_parser_page')
def resume_parser_page():
    return render_template('resumeparser.html')

@app.route('/parse_resume_and_jobdesc', methods=['POST'])
def parse_resume_route():
    f1 = request.files['resume']
    f1.save(f1.filename)
    # print("file is there",f1.filename)
    result = parse_resume(f1.filename)
    all_questions = []

    f2 = request.files['job_description']
    f2.save(f2.filename)
    job_description_parser = parseJobDescription(f2.filename)
    print(job_description_parser)
    job_desc_skillstring = ', '.join(job_description_parser)
    query1 = job_desc_skillstring + " Filter out the valid technical skills and return a list containing all the valid skills from which interview questions could be asked without mentioning any title or description."
    valid_skills_from_job_desc = answer_query(query1)
    print(valid_skills_from_job_desc)
    
    
    if(valid_skills_from_job_desc=="Failed to generate response."):
        valid_skills_from_job_desc = job_description_parser
    
    skills_string = ', '.join(skills_from_resume)
    # job_desc_skills = ', '.join(valid_skills_from_job_desc)
    query = (
    "You are an expert interview trainer. The following is a list of technical skills: "
    + skills_string +
    ". For each skill, generate 10 high-quality technical interview questions "
    "that are relevant for job interviews. "
    "Do not mention the skill name or number them. Just return all the questions in plain text, separated by new lines."
)

    print(query)
    # res = gem_model.generate_content(query)
    # text = res.text
    res = answer_query(query)
    if(res=="Failed to generate response."):
        for skill in skills_from_resume:
            query_for_web_scrapping = f"{skill} top interview questions"
            questions = scrape_questions_from_search(query_for_web_scrapping, num_results=2)
            all_questions.extend(questions)
    else:
        all_questions = extract_questions(res)
    
    # Create a DataFrame from the list of questions
    df = pd.DataFrame({'Question': all_questions})

    # Save the DataFrame to a CSV file
    csv_file_path = 'questions_csv.csv'
    df.to_csv(csv_file_path, index_label='Question Number', mode='w')
        
    return render_template('resume_parser.html', entities=result,questions=all_questions)



# @app.route('/gemini')
# def indexx():
#     return render_template('gemini.html')

# @app.route('/generate_question', methods=['POST'])
# def generate_question():
#     if request.method == 'POST':
#         user_answer = request.form['user_answer']

#         # Your gemini code
#         response = gem_model.generate_content(user_answer)

#         # Convert Markdown response to HTML
#         html_response = response.text
        
#         # print(response.text)
#         return render_template('geminiresult.html', response=html_response)







# @app.route('/scrape_questions', methods=['POST'])
# def scrape_questions():

#     skills = request.form['skills'].split(',')
#     all_questions = []

#     for skill in skills:
#         query = f"{skill} interview questions"
#         questions = scrape_questions_from_search(query, num_results=10)
#         all_questions.extend(questions)

#     return render_template('web_scrap.html', questions=all_questions)


@app.route('/uploade', methods=['POST', 'GET'])
def upload_file():
    if request.method == 'POST':
        # f.save("somefile.jpeg")
        # f = request.files['file']

        f = request.files['file'].read()
        npimg = np.frombuffer(f, dtype=np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
        face_properties = face_classifier.classify(img, face_detector, model)
        # label = face_properties[0]['label']
        # if (label == "happy" or label == "happy" or label == "happy" or label == "happy" or label == "happy" or label == "happy" )
        face_functions.update_face_analytics(face_properties)
        return json.dumps(face_properties)

@app.route('/finish', methods=['POST','GET'])
def finish():
    print("In finish")
    # return redirect(url_for('result'))

    return render_template("success.html", interview_score = utility_functions.interview_score, candidate_data=utility_functions.candidate_data)
    # return send_from_directory('templates', 'success.html')


# @app.route('/result', methods=['GET', 'POST'])
# def result():
#     return render_template("success.html")


@app.route('/interview', methods=['POST'])
def interview():
    return render_template('interview.html')

@app.route('/start', methods=['POST'])
def start():
    try:
        # start video interview from frontend 
        file = open('report_file.txt', 'r', encoding='utf-8', errors='ignore')
        data = file.readlines()
        text = []
        for line in data:
            word = line.split()
            text.append(word)

        index = 0
        for i in range(len(text)):
            if len(text[i]) > 0:
                if text[i][0] == 'Skills:':
                    index = i
                    break

        skills = text[index + 1]
        print(skills)
        options = webdriver.ChromeOptions()
        options.add_argument("headless")
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

        # driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

        count = 1
        interview_score = 0

        interview_score = utility_functions.ask_first_question(df, count, driver)
        # print(interview_score)
        
        return render_template("success.html", interview_score=interview_score, candidate_data=utility_functions.candidate_data)
    except Exception as e:
        print(e)
        return 'Internal Server Error', 500



@app.route('/updatequestion')
def updatequestion():
    if (len(utility_functions.asked) == 0):
        return jsonify("Question will be displayed here")
    question = utility_functions.asked[len(utility_functions.asked)-1]
    return jsonify(question)


@app.route('/resume')
def hello():
    return render_template('app.html')


@app.route('/success', methods=['POST'])
def success():
    if request.method == 'POST':
        f1 = request.files['file1']
        f2 = request.files['file2']
        f1.save(f1.filename)
        f2.save(f2.filename)
        print(f1.filename)
        start_file(f1.filename, f2.filename)
        return render_template("success_resume.html", name=f2.filename)

@app.route('/changetabs', methods=['POST','GET'])
def changetabs():
    utility_functions.set_question_flag()
    value = request.args.get('value')
    value = "Number of change Tabs ="+value+"\n"
    file1 = open("report_file.txt", "a")
    file1.write(value)
    file1.write("\n")
    file1.close()


    return jsonify("done")

if __name__ == '__main__':

    # Run the flask app
    app.run(debug=True)


