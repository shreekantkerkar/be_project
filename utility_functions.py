import pandas as pd
import numpy as np
import random
from sentence_transformers import SentenceTransformer, util
import random
import pyaudio
import wave
import speech_recognition as sr

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import os
from webdriver_manager.chrome import ChromeDriverManager
from sentence_transformers import SentenceTransformer, util
import requests

from gtts import gTTS
import os
import playsound
import random
from pygame import mixer 
from speech_classifier import audio_classifier
import face_functions
import pyttsx3

count = 0
interview_score = 0
asked = []

flag = False
# global question_flag 
question_flag = True
# analyticsDict = {
#   "angry": 0,
#   "disgust": 0,
#   "fear": 0,
#   "happy": 0,
#   "sad": 0,
#   "surprise": 0,
#   "neutral": 0
# }


def set_question_flag():
    global question_flag
    question_flag = False

def text_to_speech(Text):
    global count
    # # Text.replace(Text[0], 'Next ') #put where this function is called
    # # Text.replace(Text[1], 'Question')
    Text = Text[2:]
    tts = 'tts'
    tts = gTTS(text=Text, lang = 'en')
    file1 = str("hello" + str(count) + ".mp3")
    tts.save(file1)
    # playsound.playsound(file1,True)
    # tts = gTTS(Text)
    # tts.save("hello.mp3")
    mixer.init()
    mixer.music.load(file1)
    mixer.music.play()

    while mixer.music.get_busy() == True: #this is to make sure the audio is playing 
        continue
    flag = True
    # print('after')
    # # os.remove(file1)
    # return
    # Text = Text[2:]

    # engine = pyttsx3.init()
    # engine.setProperty('rate', 150)
    # engine.say(Text)
    # engine.runAndWait()


def speech_to_text():
    chunk = 1024
    sample_format = pyaudio.paInt16
    channels = 1
    fs = 44100  # frames per channel
    seconds = 5

    filename = "candidate_answer.wav"

    p = pyaudio.PyAudio()

    print("Speech Recording ...")

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    print("... Ending Speech Recording")
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))
        wf.close()

    r = sr.Recognizer()

    savedAudio = sr.AudioFile('candidate_answer.wav')

    with savedAudio as source:
        audio = r.record(source)
    try:
        s = r.recognize_google(audio)
        return str(s)
    except Exception as e:
        print("Exception: " + str(e))
        return str(e)


def context(string1, string2):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentences1 = [string1]
    sentences2 = [string2]

    embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    # print(string1, string2, cosine_scores[(0, 0)])
    # print(string1, string2, cosine_scores[0][0])
    print(cosine_scores[0][0])
    return cosine_scores[0][0]


def start_driver():
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()))


def next_question(similarity, x, data, driver):
    global count
    count += 1
    global interview_score
    subjects = list(data['Subject'].unique())

    if similarity*100 > 75:
        difficulty = 3
        subject = x['Subject'].to_string()
        interview_score += 9
        ask_question(data, difficulty, subject, driver)
    elif 50 < similarity*100 <= 75:
        difficulty = 2
        subject = subjects[random.randint(0, len(subjects)-1)]
        interview_score += 6
        ask_question(data, difficulty, subject, driver)
    elif 25 < similarity*100 <= 50:
        difficulty = 1
        subject = x['Subject'].to_string()
        interview_score += 4
        ask_question(data, difficulty, subject, driver)
    else:
        difficulty = 1
        subject = x['Subject'].to_string()
        # skill_question(x, count, data)
        skill_question(data, difficulty, subject, driver)


def ask_first_question(data, c, driver):
    global count
    count = c
    x = data.sample()
    print(x)
    first_question = x['Question'].to_string()
    first_answer = x['Actual Answer'].to_string()
    if first_question in asked:
        ask_first_question(data, driver)
    print(first_question)
    asked.append(first_question)
    write_to_file(first_question)
    text_to_speech(first_question)  # ask question
    candidate_response = speech_to_text()
    write_to_file(candidate_response)
    #speech is recorded and converted to text
    #classify the speech using the audio_classifier()
    audio_properties = audio_classifier() 
    print(candidate_response)
    similarity = context(first_answer, candidate_response)
    write_to_file(str(similarity))
    #! call finish_file
    face_functions.append_face_to_file()
    next_question(similarity, x, data, driver)

    return interview_score


def ask_question(data, difficulty, subject, driver):
    global interview_score, question_flag
    if question_flag == False:
        return interview_score
    print(count)
    if count>5:
        return interview_score
    else:
        temp_data = data[(data['Difficulty'] == difficulty)]
        x = temp_data.sample()
        question = x['Question'].to_string()
        if question in asked:
            ask_question(data, difficulty, subject, driver)
        print(question)
        asked.append(question)
        write_to_file(question)
        text_to_speech(question)  # ask question

        actual_answer = x['Actual Answer'].to_string()

        candidate_response = speech_to_text()
        print(candidate_response)
        write_to_file(candidate_response)
        audio_properties = audio_classifier()
        similarity = context(actual_answer, candidate_response)
        write_to_file(str(similarity))
        print(similarity)
        face_functions.append_face_to_file()
        next_question(similarity, x, data, driver)


def skill_question(df, difficulty, subject, driver):
    global interview_score, question_flag, count

    if question_flag == False:
        return interview_score
    if count > 5:
        return interview_score
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
    # print(skills)
    final_questions = []
    final_answers = []
    subject = skills[random.randint(0, len(skills)-1)]
    print(subject)
    # subject = 'Python'
    url = f"https://www.interviewbit.com/{subject}-interview-questions/"
    val = url
    print(url)
    response = requests.get(url)
    if response.status_code == 200:
        wait = WebDriverWait(driver, 10)
        driver.get(val)
        get_url = driver.current_url
        wait.until(EC.url_to_be(val))
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, features="html.parser")

        articles = soup.find_all('h3')
        questions = []
        for article in articles:
            questions.append(article.get_text())
        final_questions = []
        for i in questions:
            if 48 <= ord(i[0]) <= 57:
                final_questions.append(i)

        answers = soup.find_all('article', attrs={'class': 'ibpage-article'})
        final_answers = []
        for answer in answers:
            final_answers.append(answer.get_text())
        for i in range(len(final_answers)):
            final_answers[i] = final_answers[i].replace("\n", " ")
        print(final_questions)
        index = random.randint(0, len(final_questions))
        question = final_questions[index]
        if question not in asked:
            print(subject, " - ", question)
            asked.append(question)
            write_to_file(question)
            text_to_speech(question)
            actual_answer = final_answers[index]
            print(actual_answer)
            # get anscandidate_response
            candidate_response = speech_to_text()
            print(candidate_response)
            write_to_file(candidate_response)
            audio_properties = audio_classifier()
            similarity = context(actual_answer, candidate_response)
            write_to_file(str(similarity))
            x = df.sample()
            next_question(similarity, x, df, driver)

    else:
        print(subject, "Website not found")
        ask_question(df, difficulty, subject, driver)


def write_to_file(data):
    file = open("report_file.txt", "a")
    file.write(data)
    file.write("\n")
    file.close()


# def finish_file():
#     print(analyticsDict)
#     confidence = 0
#     nervous = 0
#     neutral = 0

#     for key in analyticsDict:
#         if key == "happy":
#             confidence += analyticsDict[key]
#         if key == "sad":
#             nervous += analyticsDict[key]
#         if key == "angry":
#             nervous += analyticsDict[key]
#         if key == "neutral":
#             neutral += analyticsDict[key]
#         if key == "disgust":
#             nervous += analyticsDict[key]
#         if key == "fear":
#             nervous += analyticsDict[key]
#         if key == "surprise":
#             nervous += analyticsDict[key]
    
#     total = confidence + nervous + neutral
#     if(total != 0):
#         confidence = ((confidence + neutral/2)/total) * 100
#         neutral = (neutral/total) * 100
#         nervous = (nervous /total) * 100

#     mainstr = "confidence: " + str(confidence) + "%\t nervousness: " + str(nervous) + "%\t neutral: " + str(neutral)
#     print(mainstr)  
#     print(analyticsDict)

#     file1 = open("face_result.txt", "a")
#     file1.write(mainstr)
#     file1.write("\n")
#     file1.close()

#     for key in analyticsDict:
#         analyticsDict[key] = 0
#     print(analyticsDict)
    
#     return "done"

# def update_face_analytics(face_properties):
#     if (len(face_properties)!=0 ):
#             analyticsDict[face_properties[0]['label']] +=1
        
