from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status
from .serializers import UploadedFileSerializer  # Import the serializer
from api import extractor
from api import bb_image
import json
import cv2
import base64
from .models import UploadedFile

@api_view(['POST'])
def extract_information(request):
    if request.method == 'POST':
        file = request.FILES.get('file')  # Use get() to avoid KeyError
        if file:
            serialized_data = {'file': file}
            serializer = UploadedFileSerializer(data=serialized_data)
            if serializer.is_valid():
                uploaded_file_instance = serializer.save()  # Save the file instance to the database

                file_text = extractor.text_extractor(file)
                file_path = file.name
                print("File Name: ",file_path)
                
                # Get the file path of the uploaded file instance
                file_path_img = uploaded_file_instance.file.path
                print("Image file path: ", file_path_img)
                
                docs = extractor.create_haystack_document(file_text, file_path)
                preprocessed_docs = extractor.preprocess_documents(docs)
                retriever = extractor.create_bm25_retriever(preprocessed_docs, top_k=2)
                qa_template = extractor.qa_template
                prompt_node = extractor.create_prompt_node(
                    model_name_or_path="mistralai/Mixtral-8x7B-Instruct-v0.1",
                    api_key=extractor.HF_TOKEN,
                    qa_template=qa_template,
                    max_length=500,
                    model_max_length=5000
                )
                
                rag_pipeline = extractor.create_rag_pipeline(retriever, prompt_node)
                
                question_list = extractor.create_question_list()
                
                response = extractor.extract_answers(question_list, rag_pipeline)
                
                extracted_strings = extractor.extract_substrings(response)
                
                extracted_strings_json = extractor.string_to_json(extracted_strings)
                
                json_string = extracted_strings_json
                
                cleaned_data = json.loads(json_string)
                
                extractor.save_extracted_strings_to_json(cleaned_data, output_folder='output')
                
                # Process the image with bounding boxes
                processed_image = bb_image.process_image_with_word_bounding_boxes(file_path_img)
                
                # Check if the processed image is valid before encoding
                # Read the image file
                with open('processed/processed_image_with_word_boxes.jpg', 'rb') as f:
                    image_data = f.read()
                
                # Encode the image data in base64
                encoded_image = base64.b64encode(image_data).decode('utf-8')

                # Create a JSON response with the encoded image
                response_image_data = {'image': encoded_image}
                
                # Return both the cleaned data and the processed image
                return Response({"cleaned_data": cleaned_data, "processed_image": response_image_data})
            else:
                return Response({"error": "Failed to process the image."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    else:
        return Response({"error": "No file provided."}, status=status.HTTP_400_BAD_REQUEST)

