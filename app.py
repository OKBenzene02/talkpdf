import os
from flask import Flask, request, render_template, Response, session, jsonify
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Ollama

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = 'therandomstring'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

load_dotenv()
google_api = os.getenv('GOOGLE_API_KEY')

# Global variables to store the processed PDF data
vector_store = None
chat_history = []

def clean_pages(pages):
    def clean_text(text):
        return text.encode('ascii', 'ignore').decode('ascii')

    for i, doc in enumerate(pages):
        pages[i].page_content = clean_text(doc.page_content)

    return pages

def get_llm_model(model_name='gemini'):
    if model_name == 'gemini':
        return GoogleGenerativeAI(google_api_key=google_api, model='gemini-pro')
    elif model_name == 'mistral':
        return Ollama(model='mistral')
    elif model_name == 'llama3.2':
        return Ollama(model='llama3.2')
    else:
        # Default to Gemini
        return GoogleGenerativeAI(google_api_key=google_api, model='gemini-pro')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/update_model', methods=['POST'])
def update_model():
    data = request.get_json()
    model = data.get('model')
    session['current_model'] = model
    print(f"Model switched to: {model}") 
    return jsonify({'status': 'success'})

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    global vector_store, chat_history
    chat_history = []
    vector_store = None

    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']
    if file.filename == '' or not file.filename.endswith('.pdf'):
        return 'Invalid file', 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Process the PDF
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()

    pages = clean_pages(pages)

    text_splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    docs = text_splits.split_documents(pages)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma.from_documents(documents=docs, embedding=embeddings)

    # Summarize the PDF content
    google_model = GoogleGenerativeAI(google_api_key=google_api, model='gemini-pro')
    
    # Combine the page content into a single string for summarization
    combined_text = "\n".join([doc.page_content for doc in docs])
    
    # Create a summary of the combined document text
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "Please provide a brief summary of the following document."),
        ("human", combined_text)  # Pass the combined text directly
    ])

    summary_response = summary_prompt | google_model | StrOutputParser()
    
    # Store the summary in the chat history
    chat_history.append(AIMessage(content=f"Summary of the PDF: {summary_response}"))

    return 'PDF uploaded and processed successfully', 200

@app.route('/chat', methods=['POST'])
def chat():
    global vector_store, chat_history
    
    question = request.form.get('question')  # Corrected to handle form data
    model_name = session.get('current_model', 'gemini')  # Default to gemini if not set

    if not vector_store:
        return 'Please upload a PDF first', 400

    # Get the appropriate model
    model = get_llm_model(model_name)
    out_parse = StrOutputParser()
    retriever = vector_store.as_retriever()

    instruction_to_system = """
    You are an AI assistant specialized in providing in-depth answers about a PDF document. When a user asks a question, analyze the provided context and any previously answered questions to deliver comprehensive, detailed responses. Aim to include relevant examples, explanations, and connections to enhance the user's understanding. If you cannot find the answer in the context, be transparent about it.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", instruction_to_system),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
        ("system", "Relevant context: {context}")
    ])

    chain = (
        {"context": retriever, "question": RunnablePassthrough(), "chat_history": lambda _: chat_history}
        | prompt
        | model
        | out_parse
    )

    response = chain.invoke(question)
    
    chat_history.extend([HumanMessage(content=question), AIMessage(content=response)])

    return Response(response, mimetype='text/plain')

if __name__ == '__main__':
    app.run(debug=True)