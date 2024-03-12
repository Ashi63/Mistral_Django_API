from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status
from .serializers import UploadedFileSerializer  # Import the serializer
from api import extractor
import json


@api_view(['POST'])
def extract_information(request):
    if request.method == 'POST':
        file = request.FILES.get('file')  # Use get() to avoid KeyError
        if file:
            # Assuming UploadedFile model has a 'text' field to store the extracted text
            # You may need to adjust this according to your model structure
            serialized_data = {'file': file}
            serializer = UploadedFileSerializer(data=serialized_data)
            if serializer.is_valid():
                serializer.save()  # Save the file instance to the database

                # Now you can access the data from the serializer instance
                file_text = extractor.text_extractor(file)
                
                #
                file_path = file.name
                
                # 
                docs = extractor.create_haystack_document(file_text,file_path)
                
                #
                preprocessed_docs = extractor.preprocess_documents(docs)
                
                #
                retriever = extractor.create_bm25_retriever(preprocessed_docs, top_k=2)
                
                #
                qa_template=extractor.qa_template
                
                #
                prompt_node = extractor.create_prompt_node(
                    model_name_or_path="mistralai/Mixtral-8x7B-Instruct-v0.1",
                    api_key=extractor.HF_TOKEN,
                    qa_template=qa_template,
                    max_length=500,
                    model_max_length=5000
                )
                
                #
                rag_pipeline = extractor.create_rag_pipeline(retriever, prompt_node)
                
                #
                question_list = extractor.create_question_list()
                
                #
                response = extractor.extract_answers(question_list, rag_pipeline)
                
                #
                extracted_strings = extractor.extract_substrings(response)
                
                #
                extracted_strings_json = extractor.string_to_json(extracted_strings)
                
                #
                json_string = extracted_strings_json
                # Parse the JSON string
                cleaned_data = json.loads(json_string)
                
                #
                extractor.save_extracted_strings_to_json(cleaned_data, output_folder='output')
                
                return Response(cleaned_data)
            else:
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            
        else:
            return Response({"error": "No file provided."}, status=status.HTTP_400_BAD_REQUEST)

