file_text = text_extractor(file_path)

docs = create_haystack_document(file_text)

preprocessed_docs = preprocess_documents(docs)

retriever = create_bm25_retriever(preprocessed_docs, top_k=2)

# Example usage:
prompt_node = create_prompt_node(
    model_name_or_path="mistralai/Mixtral-8x7B-Instruct-v0.1",
    api_key=HF_TOKEN,
    qa_template=qa_template,
    max_length=500,
    model_max_length=5000
)


rag_pipeline = create_rag_pipeline(retriever, prompt_node)

# Example usage
question_list = create_question_list()


# Example usage
response = extract_answers(question_list, rag_pipeline)

# Example usage:
extracted_strings = extract_substrings(response)

extracted_strings_json = string_to_json(extracted_strings)


# Example usage:
output_folder = "output"
save_extracted_strings_to_json(extracted_strings_json, output_folder)