

# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

"""
Installing the dependencies 

#pip install PyPDF2
#pip install transformers

"""

# Necessary Libraries
import numpy as np 
import pandas as pd 
import os
import PyPDF2 #For extracting data from PDF

pdf_path = '/kaggle/input/pdf-file/NLP using Transformer.pdf' 

#Function to extract data from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text
text = extract_text_from_pdf(pdf_path)
lower_text = text.lower()

print(lower_text)

abstract_index = lower_text.find("abstract")
references_index = lower_text.find("references")

#Removing the Metadata from the text
if abstract_index != -1:
    if references_index != -1:
        main_content = text[abstract_index + len("abstract"):references_index].strip()
    else:
        # In case 'References' is not found, take everything after 'Abstract'
        main_content = text[abstract_index + len("abstract"):].strip()

    print("Main Content:\n", main_content)
else:
    print("The 'Abstract' section was not found in the text.")

from transformers import pipeline

summarizer = pipeline('summarization',model="google/t5-base")

max_chunk_size = 1024 #maximum number of tokens allowed by the model


"""function to forms chunks upto size of the max_tokens"""
def chunks_up(main_content, max_chunk_size):
   
    chunks = []
    for i in range(0, len(main_content), max_chunk_size):
        chunks.append(main_content[i:i + max_chunk_size])
    return chunks

def combine_chunks(main_content, max_chunk_size):
    return chunks_up(main_content, max_chunk_size)


"""Forming the summary for the whole document"""

def combine_summary(chunks, max_chunk_size):
    summary = []
    for chunk in chunks:
        s = summarizer(chunk, max_length=150, min_length=10)
        summary.append(s[0]['summary_text'])  # Use the correct key for summary
    combined_summary = ' '.join(summary)
    return combined_summary


#function calls 
chunks = combine_chunks(main_content, max_chunk_size)
result = combine_summary(chunks, max_chunk_size)
print(result)