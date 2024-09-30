import typer
import fitz  # pymupdf
import json
import os
import hashlib
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from groq import Groq
import numpy as np
import vertexai
from rich import print
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

app = typer.Typer()
console = Console()

# Global variables
embedded_chunks = []
embedding_model = None
llm_client = None
model_name = "gemma2-9b-it"
CONFIG = json.load(open("config.json"))

vertexai.init(project=CONFIG.get("project_id"), location=CONFIG.get("region"))


def calculate_md5(file_path: str) -> str:
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def extract_text_chunks(
    file_path: str, chunk_size: int = 400, overlap: int = 100
) -> List[str]:
    """Extract text chunks from a file."""
    doc = fitz.open(file_path)
    chunks = []
    text = ""
    for page in doc:
        text += page.get_text()

    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i : i + chunk_size]
        chunks.append(chunk)

    return chunks


def embed_text(
    texts: List[str],
    task: str = "RETRIEVAL_DOCUMENT",
    model_name: str = "text-multilingual-embedding-002",
) -> List[List[float]]:
    """Embed texts using Vertex AI embedding model, processing 5 texts at a time."""
    global embedding_model
    if embedding_model is None:
        embedding_model = TextEmbeddingModel.from_pretrained(model_name)

    all_embeddings = []
    for i in range(0, len(texts), 5):
        batch = texts[i : i + 5]
        inputs = [TextEmbeddingInput(text, task) for text in batch]
        embeddings = embedding_model.get_embeddings(inputs)
        all_embeddings.extend([embedding.values for embedding in embeddings])

    return all_embeddings


def process_chunks(chunks: List[str]) -> List[dict]:
    """Process chunks by embedding them."""
    embeddings = []
    for i in range(0, len(chunks), 5):
        batch = chunks[i : i + 5]
        embeddings.extend(embed_text(batch))

    return [
        {"text": chunk, "embedding": embedding}
        for chunk, embedding in zip(chunks, embeddings)
    ]


def save_embedded_chunks(chunks: List[dict], file_path: str):
    """Save embedded chunks to a JSON file."""
    with open(file_path, "w") as f:
        json.dump(chunks, f)


def load_embedded_chunks(file_path: str) -> List[dict]:
    """Load embedded chunks from a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def find_similar_chunks(query_embedding: List[float], top_k: int = 3) -> List[str]:
    """Find the most similar chunks to the query."""
    similarities = [
        cosine_similarity(query_embedding, chunk["embedding"])
        for chunk in embedded_chunks
    ]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [embedded_chunks[i]["text"] for i in top_indices]


def get_llm_response(prompt: str, model_name: str) -> str:
    """Get response from LLM using Groq API."""
    global llm_client
    if llm_client is None:
        llm_client = Groq(api_key=CONFIG.get("groq_key"))

    chat_completion = llm_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model_name,
    )
    return chat_completion.choices[0].message.content


def get_embedding_file_path(file_path: str) -> str:
    """Generate the embedding file path based on the file's MD5 hash."""
    file_md5 = calculate_md5(file_path)
    return f"embedded_chunks_{file_md5}.json"


@app.command()
def initialize(file_path: str):
    """Initialize the application by processing the file and embedding chunks."""
    global embedded_chunks

    console.print(f"[bold blue]Initializing with file:[/bold blue] {file_path}")

    embedding_file = get_embedding_file_path(file_path)

    if os.path.exists(embedding_file):
        console.print(
            f"[bold green]Existing embedding file found. Loading...[/bold green]"
        )
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("Loading existing embeddings...", total=None)
            embedded_chunks = load_embedded_chunks(embedding_file)
            progress.update(task, description="Loaded existing embeddings")
        console.print(
            "[bold green]Loaded existing embeddings. You can now start asking questions.[/bold green]"
        )
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("Extracting text chunks...", total=None)
        chunks = extract_text_chunks(file_path)
        progress.update(
            task, description=f"Extracted {len(chunks)} chunks from the file"
        )

        task = progress.add_task("Processing chunks...", total=None)
        embedded_chunks = process_chunks(chunks)
        progress.update(task, description="Embedded all chunks")

        task = progress.add_task("Saving embedded chunks...", total=None)
        save_embedded_chunks(embedded_chunks, embedding_file)
        progress.update(task, description=f"Saved embedded chunks to {embedding_file}")

    console.print(
        "[bold green]Initialization complete. You can now start asking questions.[/bold green]"
    )


@app.command()
def chat(question: str):
    """Chat with the application."""
    global embedded_chunks

    if not embedded_chunks:
        console.print(
            "[bold red]Please initialize the application with a file first.[/bold red]"
        )
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("Embedding question...", total=None)
        question_embedding = embed_text([question])[0]
        progress.update(task, description="Question embedded")

        task = progress.add_task("Finding similar chunks...", total=None)
        similar_chunks = find_similar_chunks(question_embedding)
        progress.update(task, description="Similar chunks found")

        task = progress.add_task("Generating response...", total=None)
        context = "\n".join(similar_chunks)
        prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        response = get_llm_response(prompt, model_name)
        progress.update(task, description="Response generated")

    console.print(
        Panel(
            f"[bold green]Answer:[/bold green] {response}",
            title="Response",
            expand=False,
        )
    )


def interactive():
    """Run the application in interactive mode."""
    global embedded_chunks

    console.print("[bold cyan]Welcome to the Interactive File Q&A System![/bold cyan]")

    while True:
        if not embedded_chunks:
            file_path = Prompt.ask(
                "Please enter the path to your file (or 'q' to quit)"
            )
            if file_path.lower() == "q":
                break

            if not os.path.exists(file_path):
                console.print("[bold red]File not found. Please try again.[/bold red]")
                continue

            initialize(file_path)

        console.print("\n[bold cyan]Enter your question below[/bold cyan]")
        console.print("[dim](or 'q' to quit, 'c' to change file)[/dim]")
        question = Prompt.ask("Question")

        if question.lower() == "q":
            break
        elif question.lower() == "c":
            embedded_chunks = []
            continue

        chat(question)
        console.print("\n[dim]---[/dim]\n")

    console.print(
        "[bold cyan]Thank you for using the Interactive File Q&A System. Goodbye![/bold cyan]"
    )


if __name__ == "__main__":
    typer.run(interactive)
