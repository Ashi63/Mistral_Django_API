import PyPDF2
from PIL import Image
import pytesseract
import json
import os
from haystack import Document
from haystack import Pipeline
from haystack.nodes import BM25Retriever
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import PreProcessor,PromptModel,PromptTemplate,PromptNode

# Path for Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Hugging Face API token
HF_TOKEN = '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'

# function will extract text from the file 
def text_extractor(file):
    filename = file.name  # Get the name of the uploaded file
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        with Image.open(file) as img:
            text = pytesseract.image_to_string(img)
    elif filename.endswith('.pdf'):
        text = ""
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    else:
        raise ValueError("Unsupported file format")
    
    return text

# function to create a haystack document
def create_haystack_document(file_text,file_path):
    """
    Create a Haystack Document from a file.

    Args:
        file_path (str): Path to the file.

    Returns:
        Document: Haystack Document object.
    """
    doc = Document(content=file_text,meta={"file_path": file_path})
    docs = [doc]
    return docs
    
# function to create proprocessed documents
def preprocess_documents(docs):
    """
    Preprocesses a list of documents using Haystack's PreProcessor.

    Args:
        docs (List[Document]): List of Document objects to be preprocessed.

    Returns:
        List[Document]: List of preprocessed Document objects.
    """
    processor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=True,
        split_by="word",
        split_length=500,
        split_respect_sentence_boundary=True,
        split_overlap=0,
    )
    preprocessed_docs = processor.process(docs)
    return preprocessed_docs

# function to create bm25 retriever
def create_bm25_retriever(preprocessed_docs,top_k: int = 2):
    """
    Create a BM25 retriever using the specified document store.

    Args:
        document_store (BaseDocumentStore): Document store object.
        top_k (int): Number of documents to return as top results.

    Returns:
        BM25Retriever: BM25 retriever object.
    """
    document_store = InMemoryDocumentStore(use_bm25=True)
    document_store.write_documents(preprocessed_docs)
    retriever = BM25Retriever(document_store, top_k=top_k)
    return retriever

# Define the QA template
qa_template = PromptTemplate(prompt=
    """Using exclusively the information contained in the context, answer only the question asked without adding
    suggestions for possible questions, and respond exclusively in English. If the answer cannot be deduced from the
    context, respond: "Not sure because not relevant to the context."
    Context: {join(documents)};
    Question: {query}
    """
)

# function to create prompt node
def create_prompt_node(model_name_or_path, api_key, qa_template, max_length=500, model_max_length=5000):
    """
    Create a PromptNode with the specified configuration.

    Args:
        model_name_or_path (str): Name or path of the prompt-based model.
        api_key (str): API key for accessing the model.
        qa_template (PromptTemplate): PromptTemplate object defining the QA prompt.
        max_length (int): Maximum length of generated prompt.
        model_max_length (int): Maximum length of the model input.

    Returns:
        PromptNode: PromptNode object.
    """
    prompt_node = PromptNode(
        model_name_or_path=model_name_or_path,
        api_key=api_key,
        default_prompt_template=qa_template,
        max_length=max_length,
        model_kwargs={"model_max_length": model_max_length}
    )
    return prompt_node

# function to create a rag pipeline
def create_rag_pipeline(retriever, prompt_node):
    """
    Create a Haystack pipeline with retriever and prompt node components.

    Args:
        retriever: Retriever component (e.g., BM25Retriever).
        prompt_node: PromptNode component.

    Returns:
        Pipeline: Haystack Pipeline object.
    """
    rag_pipeline = Pipeline()
    rag_pipeline.add_node(component=retriever, name="retriever", inputs=["Query"])
    rag_pipeline.add_node(component=prompt_node, name="prompt_node", inputs=["retriever"])
    return rag_pipeline

# function for list of question
def create_question_list():
    """
    Create a list of questions to be asked.

    Returns:
        List[str]: List of questions.
    """
    question_list = [
        'What is the Invoice number?',
        'What is Invoice Date of Issue?',
        'What is the name of Seller?',
        'What is the seller Tax Id number?',
        'What is the address of Seller?',
        'What is the name of Client?',
        'What is the Client Tax Id number?',
        'What is address of Client?',
        'What is the IBAN?',
        #'Description of first item?',
        #'What is quantity of first item?',
        #'What is UM of first item?',
        #'What is Net price of first item?',
        #'What is VAT percentage of first item?',
        #'What is Gross worth of first item?',
        'What is VAT percentage in Summary?',
        'What is Net worth amount in Summary?',
        'What is Vat Amount in Summary?',
        'What is Gross Worth amount?',
    ]    
    return question_list

# function to extract real text from file
def extract_answers(question_list, rag_pipeline):
    """
    Extract answers to a list of questions using a Haystack pipeline.

    Args:
        question_list (List[str]): List of questions.
        rag_pipeline (Pipeline): Haystack Pipeline object.

    Returns:
        Dict[str, str]: Dictionary mapping questions to answers.
    """
    response = {}
    for question in question_list:
        answer = rag_pipeline.run(query=question)
        response[question] = answer["results"][0].strip()
    return response

# function to extract substrings form response
def extract_substrings(response):
    """
    Extract substrings from the answers in the response dictionary and remove trailing period.

    Args:
        response (Dict[str, str]): Dictionary mapping questions to answers.

    Returns:
        Dict[str, str]: Dictionary mapping questions to extracted substrings.
    """
    extracted_strings = {}
    for question, answer in response.items():
        index_of_is = answer.find('is')
        if index_of_is != -1:
            extracted_string = answer[index_of_is + len('is'):].strip()
            if extracted_string.endswith('.'):
                extracted_string = extracted_string[:-1]  # Remove last character (period)
            extracted_strings[question] = extracted_string
    return extracted_strings

# Convert the extracted strings into JSON
def string_to_json(string):
    extracted_strings_json = json.dumps(string)
    return extracted_strings_json

# function to create a json file.
def save_extracted_strings_to_json(extracted_strings, output_folder):
    """
    Save the extracted strings to a JSON file inside the output folder.

    Args:
        extracted_strings (Dict[str, str]): Dictionary mapping questions to extracted substrings.
        output_folder (str): Path to the output folder.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Define the path to the output JSON file
    output_file_path = os.path.join(output_folder, "extracted_strings.json")

    # Write the extracted strings to the JSON file
    with open(output_file_path, "w") as f:
        json.dump(extracted_strings, f, indent=4)

