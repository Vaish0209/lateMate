from flask import Flask, render_template, request
from pythoncode import Summarization
from pythoncode import QB
from pythoncode import Question_answering
from pythoncode import Quiz
from pythoncode import Excel_file_info
from pythoncode import get_arxiv_paper_introduction
from pythoncode import paper_summarization
from pythoncode import recommender
from pythoncode import visitor_body
from pythoncode import remove_citations
from pythoncode import remove_after_references
from pythoncode import wrap_text
from pythoncode import write_string_to_file
from pythoncode import extract_filename


from googletrans import Translator
import tempfile
from langchain.tools import DuckDuckGoSearchRun
import speech_recognition as sr


app = Flask(__name__)


@app.route("/")
@app.route("/index.html")
def home():
    return render_template('index.html')


@app.route("/summarization.html")
def summarization():
    return render_template('summarization.html')

@app.route("/question_ans.html")
def question_ans():
    return render_template('question_ans.html')

@app.route("/questionbank.html")
def questionbank():
    return render_template('questionbank.html')

@app.route("/quiz.html")
def quiz():
    return render_template('quiz.html')


@app.route("/GoogleTranslate.html")
def GoogleTranslate():
    return render_template('GoogleTranslate.html')

@app.route("/excel.html")
def excel():
    return render_template('excel.html')

@app.route("/research_paper_preview.html")
def research_paper_preview():
    return render_template('research_paper_preview.html')


@app.route("/research_paper_summary.html")
def research_paper_summary():
    return render_template('research_paper_summary.html')


@app.route("/references_using_keywords.html")
def references_using_keywords():
    return render_template('references_using_keywords.html')

@app.route("/process_pdf", methods=['POST'])
def process_pdf():
    pdf_file = request.files['pdf_file']
    source = request.form['source']

    # Call the summarization and QB functions with the PDF path
    summarization_result = Summarization(pdf_file, source, points=20)

    # Render the summarization.html template and pass the results as variables
    return render_template('summarization.html', summarization_result=summarization_result)


@app.route('/answer', methods=['POST'])
def answer():
    # Get the PDF file and user question from the form
    pdf_file = request.files['pdf']
    user_question = request.form['question']

    # Check if PDF file and question are provided
    if pdf_file and user_question:
        # Call the question answering function
        response = Question_answering(pdf_file, user_question)

        # Pass the response to the HTML template
        return render_template('question_ans.html',question=user_question, response=response)

    # Handle case where PDF file or question is missing
    return render_template('question_ans.html', error='Please provide both a PDF file and a question.')



@app.route("/questionbank_creation", methods=['POST'])
def questionbank_creation():
    pdf_file = request.files['pdf_file']

    # Call the summarization and QB functions with the PDF path
    qb_result = QB(pdf_file, 'law book', points=10)

    # Render the summarization.html template and pass the results as variables
    return render_template('questionbank.html', qb_result=qb_result)


@app.route("/quiz_time", methods=['POST'])
def quiz_time():
    pdf_file = request.files['pdf_file']

    # Get the source and number of questions from the form
    source = request.form['source']
    num_questions = int(request.form['num_questions'])

    # Call the Quiz function with the PDF file, source, and number of questions
    quiz_result = Quiz(pdf_file, source, questions=num_questions)

    # Render the quiz.html template and pass the quiz_result as a variable
    return render_template('quiz.html', quiz_result=quiz_result)


@app.route('/info_extract', methods=['POST'])
def info_extract():
    # Get the PDF file and user question from the form
    pdf_file = request.files['CSV']
    user_question = request.form['question']

    response = Excel_file_info(pdf_file, user_question)
    return render_template('excel.html',question=user_question, response=response)



@app.route('/translate', methods=['POST'])
def translate():
    text = request.form['text']
    target_language = request.form['language']

    translator = Translator()
    translation = translator.translate(text, dest=target_language)
    
    return render_template('GoogleTranslate.html',original_text=text, translation=translation.text)


@app.route('/paper_preview', methods=['POST'])
def paper_preview():
    paper_name = request.form['paper_name']
    introduction = get_arxiv_paper_introduction(paper_name)
    return render_template('research_paper_preview.html', introduction=introduction)


@app.route('/paper_summary', methods=['POST'])
def paper_summary():
    pdf_file = request.files['pdf_file']
    result = paper_summarization(pdf_file)
    return render_template('research_paper_summary.html', result=result)


@app.route('/paper_recommender', methods=['POST'])
def paper_recommender():
    keywords = request.form['keywords']
    papers = recommender(keywords)
    return render_template('references_using_keywords.html', papers=papers)



if __name__ == '__main__':
    app.run(debug=True)
