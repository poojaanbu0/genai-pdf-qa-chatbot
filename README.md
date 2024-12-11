## Development of a PDF-Based Question-Answering Chatbot Using LangChain

### AIM:
To design and implement a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain, and to evaluate its effectiveness by testing its responses to diverse queries derived from the document's content.

### PROBLEM STATEMENT:
In many cases, users need specific information from large documents without manually searching through them. A question-answering chatbot can address this problem by:

1. Parsing and indexing the content of a PDF document.
2. Allowing users to ask questions in natural language.
3. Providing concise and accurate answers based on the content of the document.
  
The implementation will evaluate the chatbot’s ability to handle diverse queries and deliver accurate responses.

### DESIGN STEPS:

#### STEP 1: Environment Setup and Document Processing
- Install necessary libraries like LangChain, OpenAI, Panel, and PyPDFLoader.
- Configure the OpenAI API for embedding generation and chatbot models.
- Load PDF documents using PyPDFLoader and split the text into manageable chunks using RecursiveCharacterTextSplitter to enhance retrieval performance.
#### STEP 2: Vector Store Creation and Question Answering
- Generate embeddings from text chunks using OpenAIEmbeddings and store them in a vector database (e.g., DocArrayInMemorySearch) for similarity-based retrieval.
- Implement LangChain’s ConversationalRetrievalChain to handle user queries by retrieving relevant chunks and generating context-aware answers with OpenAI's GPT models.
#### STEP 3: Interactive Interface and Testing
- Develop an interactive GUI with Panel, featuring tabs for conversations, database queries, chat history, and configuration.
- Enable dynamic PDF loading and manage chat history for seamless interaction.
- Test the chatbot with diverse queries to validate its accuracy, efficiency, and usability.

### PROGRAM:

### OUTPUT:
![389219085-3634bf2e-7b7e-4437-a467-4fd542c2fddf](https://github.com/user-attachments/assets/b424a8c6-bd1c-423b-8b02-ba906816d9c8)

### RESULT:
Thus, a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain was implemented and evaluated for its effectiveness by testing its responses to diverse queries derived from the document's content successfully.
