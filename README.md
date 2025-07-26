# Data-Formatting-Tool

This project aims to develop a tool that converts various data formats into a structured format like Markdown for chatbot knowledge bases.

## What is the goal of this project?
The goal is to develop a tool that converts PDF and URL data into Markdown for retrieving more contextual and relevant text in RAG applications, and implementing AI-based methods for preprocessing, text summarization, and Q&A generation to enhance Markdown quality. 

Finally, including all these in one tool, developing a user-friendly interface with a modular design, which is deployed in a self-hosted virtual machine.

### Key Features and Tasks
- 1: PDF, URL-Markdown: Our tool converts PDF and URL into Markdown with proper table structures and OCR.
  
- 2: Q&A Generation: We generate accurate Q&A pairs from the Markdown input, which are contextually aligned and relevant to the input text

- 3: Text Summarization: Generating concise summaries without losing any information

- 4: Implementation: Using Flask to deploy the developed tool in our Virtual Machine.

## Installation

**bash**
git clone https://github.com/VKishorereddy084/Data-Formatting-Tool.git
cd Data-Formatting-Tool
pip install -r requirements.txt

#### Files

**Converters**: This folder contains four Python scripts that can convert PDF-md and preprocess the markdown output, URL-md, generate Q&A pairs from the Markdown data, and perform text summarization.
 - ***converts/PDF_to_MD.py***:  This script takes a PDF file as input and converts its content into clean, structured Markdown format. It extracts text from each page, removes unnecessary formatting, and outputs a Markdown (.md) file.
 - ***converts/URL_to_MD.py***: A web scraping and parsing script that processes a given URL, extracts internal URLS based on the request and generates readable text content from the webpage, and saves it in Markdown format.
 - ***converts/Q&A_Generation.py***: This script reads a Markdown file and generates question-answer pairs using advanced natural language processing techniques. It analyzes the content, formulates questions based on key concepts, and maps them to relevant answer snippets. The output is saved as an .md file.
 - ***converts/Summarization.py***: A script that reads long-form content (from Markdown) and summarizes it using abstractive methods. It produces a concise version of the content while preserving core ideas.


**Templates**: This folder contains two HTML templates that define the web-based user interface for our processing pipeline.
-***templates/index.html***: This HTML file serves as the main user interface of the application. It allows users to upload files, enter URLs, or interact with other tools in the pipeline. It is the entry point to the system.
-***templates/iselect_url.html***: A dedicated interface for selecting and managing multiple URLs in the URL-to-Markdown conversion pipeline. This template allows users to input a list of URLs, preview extracted content, and selectively choose which content to include in the final dataset.
    



