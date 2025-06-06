from pdfminer.high_level import extract_text
import docx2txt
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
import yake

# Download required NLTK resources at the beginning
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)


def extract_text_from_docx(docx_path):
    txt = docx2txt.process(docx_path)
    if txt:
        return txt.replace('\t', ' ')
    return None


def clean_text(text):
    text = text.replace('\n', '.')
    text = text.replace(':', '.')
    text = text.replace('\xa0', ' ')
    text = text.replace('●', ' ')
    resume = re.split("\.+", text)
    for i in resume:
        if i == " " or i == '•' or i == "  ":
            resume.remove(i)

    return resume


def get_nlp(text):
    tokenizer = RegexpTokenizer(r'\w+')
    doc = tokenizer.tokenize(text)
    return doc


def get_profile_section(doc):
    sec_profile = []
    for i in doc:
        if i.upper() == 'EDUCATION':
            break
        else:
            sec_profile.append(i)

    return sec_profile


def get_education_section(doc):
    sec_education = []
    flag = False
    for i in doc:
        if i.upper() == 'EDUCATION':
            sec_education.append(i)
            flag = True
        if 'Experience' in i or 'EXPERIENCE' in i or 'Projects' in i or 'PROJECTS' in i:
            break
        if 'Skills' in i or 'SKILLS' in i:
            break
        elif flag:
            sec_education.append(i)

    return sec_education


def ProperNounExtractor(text):
    global words
    proper_nouns = []
    print('PROPER NOUNS EXTRACTED :')

    try:
        sentences = nltk.sent_tokenize(text)
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            words = [word for word in words if word not in set(stopwords.words('english'))]
            tagged = nltk.pos_tag(words)
            for (word, tag) in tagged:
                if tag == 'NNP':  # If the word is a proper noun
                    proper_nouns.append(word)
        print(words)
        return words
    except LookupError:
        # Fallback if nltk resources still not available
        print("NLTK resources not found. Using basic tokenization.")
        words = text.split()
        print(words)
        return words


def get_experience_section(doc):
    sec_experience = []
    flag = False
    for i in doc:
        if i == 'Experience' or i == 'EXPERIENCE' or i == 'PROJECTS' or i == 'Projects':
            sec_experience.append(i)
            flag = True
        if i == 'SKILLS' or i == 'Skills':
            break
        elif flag:
            sec_experience.append(i)

    final_exp = []
    for i in sec_experience:
        final_exp.append(i)
    set1 = set(final_exp)
    final_exp.clear()
    for i in set1:
        final_exp.append(i)

    noun_str = ""
    for i in final_exp:
        noun_str += i
        noun_str += " "
    print("Experience Section: \n")
    final_exp = ProperNounExtractor(noun_str)

    return final_exp


def get_skills_section(doc):
    sec_skills = []
    flag = False
    for i in doc:
        if i == 'SKILLS' or i == 'Skills':
            sec_skills.append(i)
            flag = True
        if i == 'Extracurriculars' or i == 'EXTRACURRICULARS':
            break
        elif flag == True:
            sec_skills.append(i)

    skills_str = ""
    for i in sec_skills:
        skills_str += i
        skills_str += " "
    print("Skills section: \n")
    sec_skills = ProperNounExtractor(skills_str)

    return sec_skills


def get_requirements(doc):
    req_tokens = []
    flag = False
    for i in doc:
        if 'Requirements' in i or 'REQUIREMENTS' in i or 'skills' in i or 'SKILLS' in i:
            req_tokens.append(i)
            flag = True
        elif flag:
            req_tokens.append(i)

    noun_str = ""
    for i in req_tokens:
        noun_str += i
        noun_str += " "
    print("Requirements section: \n")
    final_exp = ProperNounExtractor(noun_str)

    return final_exp


def get_summary(text):
    kw_extractor = yake.KeywordExtractor()
    keywords = kw_extractor.extract_keywords(text)
    experience = []
    for i in keywords:
        experience.append(i[0])

    return experience


def get_extra(doc):
    sec_extracurriculars = []
    flag = False
    for i in doc:
        if i == 'Extracurriculars' or i == 'EXTRACURRICULARS':
            sec_extracurriculars.append(i)
            flag = True
        elif flag:
            sec_extracurriculars.append(i)
    return sec_extracurriculars