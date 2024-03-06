import pyperclip
from langchain.utilities import WikipediaAPIWrapper
import requests
from bs4 import BeautifulSoup
from fpdf import FPDF
import pandas as pd
import pandasai as pdai
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
import matplotlib.pyplot as plt
import openai
import base64
import io
import pickle
import os
import numpy as np
import re
import textwrap
import tiktoken
import openai
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from PyPDF2 import PdfReader
import panel as pn
import tempfile
import pyttsx3
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

OPENAI_API_KEY = "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" 
openai.api_key = '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'
os.environ['OPENAI_API_KEY'] = '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'


def Summarization(pdf, source, points):
    # extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = """"""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=5000,
            chunk_overlap=500,
            length_function=len
        )
        texts = text_splitter.split_text(text)
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    count = 0
    summary_joint = """"""
    question_bank = """"""
    while count != (len(texts) - 1):
        prompt = """"""
        for chunck in range(count, len(texts)):
            temp = prompt + texts[chunck]
            tokens = encoding.encode(temp)
            if len(tokens) < 10000:
                prompt = temp
                count = chunck
            else:
                break
        print(count)
        summary = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {"role": "system",
                 "content": "You are a summarization model that creates notes in point form for students."},
                {"role": "user", "content": f"""Following is a snippet of a {source}. Write short notes in bullet point form for this in 3000 tokens:
            {prompt}"""},
            ]

        )

        summary_joint = summary_joint + summary["choices"][0].message.content  # Check with and without split

    summary_joint_2 = """"""

    if len(encoding.encode(summary_joint)) > 10000:
        text_splitter_summary = CharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        texts_summary = text_splitter.create_documents([summary_joint])
        count_sum = 0
        summary_joint_2 = """"""
        while count_sum != (len(texts_summary) - 1):
            prompt_sum = """"""
            for chunck in range(count_sum, len(texts_summary)):
                temp_sum = prompt_sum + texts_summary[chunck].page_content
                tokens = encoding.encode(temp)
                if len(tokens) < 10000:
                    prompt = temp
                    count = chunck
                else:
                    break
            parses = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                messages=[
                    {"role": "system",
                     "content": "You are a summarization model that creates notes in point form for students."},
                    {"role": "user", "content": f"""Following is a snippet of a {source}. Write short notes in bullet point form for this in 3000 tokens:
            {prompt}"""},
                ]

            )
            summary_joint_2 = summary_joint_2 + parses["choices"][0].message.content

    else:
        summary_joint_2 = summary_joint

    summary_final_parse = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": "You are an expert in converting unstructured notes into structured notes for students"},
            {"role": "user", "content": f"""Rewrite these notes belonging to a {source} in {points} bullet points:
                {summary_joint}"""},
        ]

    )

    summary_final = summary_final_parse["choices"][0].message.content.split("\n")


    return summary_final




# Extract the text
def Question_answering(pdf, user_question):
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # Create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        #Moderation
        response = openai.Moderation.create(
        input= user_question)
        moderation_output = response["results"][0]
        if moderation_output["flagged"] == True:
            response = "The question asked is inappropriate."

        else:
            if user_question:
                docs = knowledge_base.similarity_search(user_question)

                llm = OpenAI()
                chain = load_qa_chain(llm, chain_type="stuff")
                with get_openai_callback() as cb:
                    response = chain.run(input_documents=docs, question=user_question)
                    
        return response



def QB(pdf, source, points):
    # extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = """"""
        for page in pdf_reader.pages:
            text += page.extract_text()
   
    questions = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        temperature = 0.2,
        messages=[
            {"role": "system",
             "content": "You are a Question and Answer Bank creation model"},
            {"role": "user", "content": f"""Generate a {points} questions with answers from the following snippet of the following {source}:
            {text}
            Give output in following format:
            1: <Question>
            Answer: <Answer>"""},
            ]

        )

    questions = questions["choices"][0].message.content.split("\n")

    return questions

def Quiz(pdf,source, questions):
# extract the text
  if pdf is not None:
    pdf_reader = PdfReader(pdf)
    text = """"""
    for page in pdf_reader.pages:
        text += page.extract_text()

    questions = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-16k",
    temperature = 0.2,
    messages=[
            {"role": "system", "content": """You are an expert at quiz creation."""},
            {"role": "user", "content": f"""You are be given a part of a {source} as follows:
             {text}
             Create a MCQ format quiz with answers of {questions} questions from the {source}. Only display the quiz."""},
      ]
    )

    return questions["choices"][0].message.content.split("\n")

# llm = OpenAI(api_token=OPENAI_API_KEY)


def Excel_file_info(uploaded_file, user_question):

    # Process uploaded file
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        llm = OpenAI()
        pandas_ai = PandasAI(llm)

        #Moderation
        response = openai.Moderation.create(
        input= user_question)
        moderation_output = response["results"][0]
        if moderation_output["flagged"] == True:
            response = "The question asked is inappropriate."

        else:
            if user_question:
                response = pandas_ai.run(df, prompt= user_question)
                
        return response



def get_arxiv_paper_introduction(paper_name):
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"ti:{paper_name}",
        "start": 0,
        "max_results": 1
    }

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        data = response.text
        soup = BeautifulSoup(data, "html.parser")

        entry = soup.find("entry")
        if entry is not None:
            summary = entry.find("summary")
            if summary is not None:
                introduction = summary.get_text()
                return introduction

    return None





parts = []

def visitor_body(text, cm, tm, fontDict, fontSize):
    y = tm[5]
    if y > 50 and y < 720:
       parts.append(text)

def remove_citations(text):
    split_text = re.split(r'(\[\d+\].*?(?=\[\d+\]|$))', text, flags=re.DOTALL)
    no_citations = [chunk for i, chunk in enumerate(split_text) if i % 2 == 0]
    citations = [chunk for i, chunk in enumerate(split_text) if i % 2 != 0]
    return ''.join(no_citations), ''.join(citations)

def remove_after_references(text: str) -> str:
    """Remove everything after a line containing 'References'."""
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if 'References' in line:
            return '\n'.join(lines[:i+1])
    return text

def wrap_text(text: str, width: int = 120) -> str:
    """Wrap text to a specified width."""
    return '\n'.join(textwrap.wrap(text, width))

def write_string_to_file(filename: str, text: str) -> None:
    """Write a string to a file."""
    with open(filename, 'w') as f:
        f.write(text)

def extract_filename(path_string):
    """Extract filename without extension from a path string."""
    base_name = os.path.basename(path_string)  # Get the filename with extension
    file_name_without_ext = os.path.splitext(base_name)[0]  # Remove the extension
    return file_name_without_ext

def paper_summarization(pdf):
  reader = PdfReader(pdf)
  for page in reader.pages:
    page.extract_text(visitor_text=visitor_body)

  text_body = "".join(parts)  
  text_body = remove_after_references(text_body)
  text_body = remove_citations(text_body)
  from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
  )
  from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
  )

  from langchain.chat_models import ChatOpenAI
  from langchain.chains import LLMChain

  context_template="You are a helpful AI Researcher that specializes in analysing ML, AI and LLM papers. Please use all your expertise to approach this task. Output your content in markdown format and include titles where relevant."

  system_message_prompt = SystemMessagePromptTemplate.from_template(context_template)

  human_template="Please summarize this paper focusing the key important takeaways for each section. Expand the summary on methods so they can be clearly understood. \n\n PAPER: \n\n{paper_content}"

  human_message_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template=human_template,
            input_variables=["paper_content"],
        )
  )
  
  
  chat_prompt_template = ChatPromptTemplate.from_messages([system_message_prompt,
                                                         human_message_prompt])

  chat = ChatOpenAI(model_name="gpt-3.5-turbo-16k",
                  temperature=0.2)

  summary_chain = LLMChain(llm=chat, prompt=chat_prompt_template)

  from langchain.callbacks import get_openai_callback

  with get_openai_callback() as cb:
    output = summary_chain.run(text_body)

  return wrap_text(output).split("#")  




def search_arxiv_papers(keywords):
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{keywords}",
        "max_results": 10  # Number of papers to retrieve
    }

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        data = response.text
        return data
    else:
        return None
    

def extract_paper_information(xml_data):
    papers = []
    soup = BeautifulSoup(xml_data, "xml")
    entries = soup.find_all("entry")

    for entry in entries:
        title = entry.title.text.strip()
        authors = [author.find("name").text.strip() for author in entry.find_all("author")]
        arxiv_id = entry.id.text.strip()

        papers.append({
            "title": title,
            "authors": authors,
            "arxiv_id": arxiv_id
        })

    return papers
    
def recommender(keywords):
    papers = []

    if keywords:
        xml_data = search_arxiv_papers(keywords)

        if xml_data:
            papers_info = extract_paper_information(xml_data)

            if papers_info:
                for paper in papers_info:
                    papers.append({
                        "title": paper["title"],
                        "authors": paper["authors"],
                        "arxiv_id": paper["arxiv_id"]
                    })
    
    return papers




