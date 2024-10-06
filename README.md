# AI-Based Generative Search System Using LangChain:

## Table of Contents:
* [Introduction](#introduction)
* [Objective](#objective)
* [Data Source](#data-source)
* [System Architecture](#system-architecture)
* [Steps for System Development](#steps-for-system-development)
* [Testing and Evaluation](#testing-and-evaluation) 
* [Challenges Faced and Lessons learned](#challenges-faced-and-lessons-learned) 
* [Conclusion](#conclusion) 
* [Tools and Libraries](#tools-and-libraries)
* [References](#references)
* [Contact Information](#contact-information)

## Introduction:
<div align="justify">In this repository, we use LangChain to build an AI-based generative search system for efficiently answering questions from multiple PDF documents. Policies documents such as Insurance policies are complex documents, and they contain a huge amount of information. This type of document is often written in dense legal language therefore customers and insurance agents often face challenges in finding required information quickly and accurately from these documents. Traditional search methods often struggle with efficiently extracting relevant information and fail in providing accurate answers and lead to time-consuming, inaccurate, and unreliable results.<br>

To address these issues, we will develop a comprehensive and robust AI-based generative search system which will be capable of effectively and accurately answering questions from a multiple PDF document. This innovative approach allows users to pose precise, context-aware questions and receive accurate answers directly from the text. By improving efficiency, accuracy, and accessibility, the system enhances document management and data retrieval, which may be useful in various sectors such as legal, financial, medical, and academic etc. </div>

## Objective:
<div align="justify">The main objective of this project is to develop a robust generative search system which will be capable of effectively and accurately answering questions from various insurance policy documents, i.e. multiple PDFs documents. We aim to use frameworks like LangChain to build a system which can efficiently retrieve and generate relevant responses from insurance policy documents.</div>

## Data Source:
 - Document: The project will use multiple insurance policy documents.<br>

 - File Format: The document is provided in PDF format.
  
## System Architecture:
<div align="justify">This system will utilize the Retrieval Augmented Generation (RAG) pipeline, which combines embedding, search, generative layers and advanced LLM frameworks LangChain to provide comprehensive and contextually relevant answers.<br>

### Retrieval-Augmented Generation (RAG):
RAG is the process of optimizing the output of a large language model, so it references an authoritative knowledge base outside of its training data sources before generating a response. Large Language Models (LLMs) are trained on vast volumes of data and use billions of parameters to generate original output for tasks like answering questions, translating languages, and completing sentences. It is a cost-effective approach to improving LLM output, so it remains relevant, accurate, and useful in various contexts. **RAG combines two types of models:**
- Retrieval Models: Pull data from a knowledge base
- Generative models: create the responses<br>

This combination makes RAG more powerful than a model that only generates responses. It can answer difficult questions and provide more informative responses.</div>

### LangChain and LlamaIndex:
<div align="justify"> In the domain of generative AI, LangChain and LlamaIndex are the two most popular frameworks which simplify the development of LLM-based applications. Both frameworks support integration with external tools and services, but their primary focus areas set them different.</br>

**LangChain** focuses on creating and managing complex sequences of operations through its use of chains, prompts, models, memory, agents and it is highly modular and flexible. <br>

**LlamaIndex** is a framework to build context-augmented LLM applications. Context augmentation means any use case that applies LLMs on top of our private or domain-specific data. LlamaIndex integrates external knowledge sources and databases as query engines for memory purposes for RAG-based apps. </div>

### Why LangChain for This Project?
<div align="justify"> LangChain is suitable for this project because it integrates several key components required for retrieval-augmented generation and provides a comprehensive framework for building generative search applications. It offers:</br>

- Document loaders for parsing PDFs.</br>
- Vector stores for embedding documents and querying.</br>
- Easy Integration with LLMs (Large Language Models) to generate accurate, context-driven answers.</br>
- Flexible Retrieval: Efficient mechanisms for querying and retrieving relevant information.</br>
- Prompt Management: Facilities to design and manage prompts effectively.</br>

LangChain also allows smooth integration with Hugging Face APIs for embeddings and generation which provide both flexibility and performance for building a generative search system.</div>
  
## Steps for System Development:
<div align="justify"> We will follow these steps to build a question-answering system using LangChain and the Retrieval-Augmented Generation (RAG) model.</br>

- **Step1: Mount Google Drive and Set Hugging Face API:**</br>
We will use Google Colab’s drive.mount to access files from Google Drive and Hugging Face API key from a file stored in Drive to authenticate and access models for embedding and generation.</br>

- **Step2: Import the Necessary Libraries:**</br>
We will import all necessary libraries like langchain, ChromaDB, pypdf for PDF extraction, and other required tools for building the RAG model pipeline.</br>

- **Step3: Load and Split Document:**</br>
We will load and merge the multiple PDFs document using the python library and split documents into manageable chunks.</br>

- **Step4: Create Embeddings and Vector Store:**</br>
We will create embeddings and store them in a vector database.</br>

- **Step5: Define Question and Initialize LLM:**</br>
In this step we will define question and Initialize LLM model</br>

- **Step6: Prompt Template Design:**</br>
We will create prompts for generating accurate responses.</br>

- **Step7: Create and Run Multiple QA Chain:**</br>
We will implement different chain types (basic, map-reduce, refine) to enhance response accuracy.</div>

## Testing and Evaluation:
<div align="justify"> The system will be tested against the self-designed queries and final generated answers will be evaluated.</div>

## Challenges Faced and Lessons learned:
- **Model Selection:**</br>
Performance and accuracy also depend on the model. Choosing the best models required extensive testing to balance computational efficiency with retrieval quality.</br>

- **Scalability:**</br>
Managing a large corpus of documents in vector stores while ensuring fast search results is a challenging task.</br>

- **Prompt Design:**</br>
Creating an effective prompt that guides LLM to generate accurate and context specific answers required significant testing and improvement.</br>

- **System Performance:**</br>
While processing large documents, it is difficult to ensure that the system remains efficient and scalable.

## Conclusion:
<div align="justify"> This project aims to develop a comprehensive and robust generative search system using RAG pipeline and advanced LLM frameworks. By utilizing LangChain framework, the final system is expected to accurately answer complex queries from a multiple pdf document, which can demonstrate the power of AI in automating and enhancing information retrieval.</div>

## Tools and Libraries:
- Python, version 3
- LangChain
- Hugging face
- ChromaDB
- Google Colab

## References:
- Python documentations
- Hugging face documentations
- Stack Overflow
- OpenAI
- Kaggle
- LangChain documentations

## Contact Information:
Created by https://github.com/Erkhanal - feel free to contact!
