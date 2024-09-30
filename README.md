# ChatFile with Gemma

ChatFile with Gemma is an interactive command-line application that allows users to ask questions about the content of PDF files using Gemma 2, a large language model. The application uses text embedding and similarity search to find relevant information from the document and generate answers.

## Features

- Extract text from PDF files
- Embed text chunks using Vertex AI's text embedding model
- Perform similarity search to find relevant context for questions
- Generate answers using Gemma 2 large language model via Groq API
- Interactive command-line interface with rich text formatting
- Caching of embedded chunks for faster subsequent queries

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/Julian-Cao/chat-file-with-gemma.git
   cd chat-file-with-gemma
   ```

2. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Set up your configuration:
   Create a `config.json` file in the project root with the following structure:
   ```json
   {
     "project_id": "your-google-cloud-project-id",
     "region": "your-google-cloud-region",
     "groq_key": "your-groq-api-key"
   }
   ```

## Usage

Run the application in interactive mode:

```
python main.py
```

Follow the prompts to:

1. Initialize the application with a PDF file
2. Ask questions about the content of the file
3. Change the file or quit the application

## Dependencies

- typer: For creating the command-line interface
- PyMuPDF (fitz): For extracting text from PDF files
- vertexai: For text embedding using Google's Vertex AI
- groq: For interfacing with the Groq API to access Gemma 2 LLM
- numpy: For numerical operations and similarity calculations
- rich: For enhanced console output and formatting

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
