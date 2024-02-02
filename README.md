# RAGtime Chatbot with your own files

A project showcasing the use of LLM's chat functionality enhanced with RAG from an exemplary document. 
The document used for RAG is latest (as for February 2024) financial report from ORLEN S.A. 
The file is avaiable [here](https://www.orlen.pl/content/dam/internet/orlen/pl/pl/relacje-inwestorskie/raporty-i-publikacje/sprawozdania/2023/3q2023/ORLEN_231031_2023kw3%20-%20RAPORT%20IIIQ2023.pdf.coredownload.pdf).
The usage of the report is completely profit-free and the document serves entirely as an example. It is publicly avaiable information nonetheless.

## What is it?

It's a terminal-based chat app with RAG and memory and question-cache created using Langchain. Uses the latest GPT-4-turbo-preview, but you san configure it yourself in the code. 
The app will automatically create a vector database based on a report while used for the first time. 

## Is it reliable?

For now, I wouldn't trust it with everything. There are inaccuracies comparing answers, even though the input remains intact. Yet, the purpose was primarily to show the possibilities, not to create a fully-fledged, ready to use product.

## How do I use it?

You need your own OpenAI API key set and Langchain installed. That's all. 

## Any next steps?

I plan to create a GUI with Streamlit and add more functionalities, such as adding/dropping your own document, extracting ready-to-use tables to be used in Pandas and what have you. The project has only educational value for me, so nothing too fancy.
