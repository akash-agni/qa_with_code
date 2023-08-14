"""
Main script for performing Question & Answer using a conversational retrieval chain.
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
from tempfile import TemporaryDirectory
from time import time
from typing import Any, List, Optional

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import DeepLake

logger = logging.getLogger(name="QA_PROMPT")


def run_command(cmd: str) -> None:
    """
    Execute console commands.

    Args:
        cmd (str): The command to be executed.
    """
    print("=" * 10)
    print(f"Executing: {cmd}")
    subprocess.run(cmd, check=True)
    print("=" * 10)


def clone_repo(url: str, repo_root_dir: Optional[str] = None) -> str:
    """
    Clone a repository from a given URL.

    Args:
        URL (str): The URL of the repository to clone.
        repo_root_dir (str, optional): The directory where the repository will be cloned. If not provided,
                                  a temporary directory will be created.

    Returns:
        str: The path to the cloned repository.
    """
    if not repo_root_dir:
        repo_root_dir = TemporaryDirectory(prefix="qa_with_code").name

    run_command(f"git clone {url} {repo_root_dir}")

    return root_dir


def load_creds() -> None:
    """
    Load API keys from 'creds.json' file and set environment variables.
    """
    with open("creds.json", "r+") as file:
        keys = json.loads(file.read())
    os.environ["OPENAI_API_KEY"] = keys["OPENAI_API_KEY"]
    os.environ["ACTIVELOOP_TOKEN"] = keys["ACTIVELOOP_TOKEN"]


def load_docs(root_dir: str) -> List[Any]:
    """
    Load documents from the specified directory.

    Args:
        root_dir (str): The directory containing the documents.

    Returns:
        List[Any]: A list of loaded documents.
    """
    docs = []
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            try:
                loader = TextLoader(os.path.join(dirpath, file), encoding="utf-8")
                docs.extend(loader.load_and_split())
            except Exception as e:
                print(str(e))
    return docs


def text_splitter(docs: List[Any], chunk_size: int = 1000) -> List[str]:
    """
    Split the input documents into text chunks.

    Args:
        docs (List[Any]): List of documents to be split.
        chunk_size (int): Size of each text chunk.

    Returns:
        List[str]: List of text chunks.
    """
    splits = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    texts = splits.split_documents(docs)
    return texts


def build_db(texts: Any, user_name: str, db_name: Optional[str]) -> Any:
    """
    Build a DeepLake vector store and upload texts to it.

    Args:
        texts (List[str]): List of text chunks.
        user_name (str): User's name for the vector store.
        db_name (str): Name of the vector store.

    Returns:
        Any: The constructed vector store.
    """
    embeddings = OpenAIEmbeddings(disallowed_special=())
    db = DeepLake(
        dataset_path=f"hub://{user_name}/{db_name}",
        embedding_function=embeddings,
    )
    db.add_documents(texts)
    return db


def build_qa_chain(db: Any, model_name: str) -> Any:
    """
    Build a conversational retrieval chain for Question & Answer.

    Args:
        db (Any): The vector store containing the documents.
        model_name (str): The name of the model to be used for Q&A.

    Returns:
        ConversationalRetrievalChain: The constructed Q&A chain.
    """

    retriever = db.as_retriever()
    model = ChatOpenAI(model_name=model_name)  # switch to 'gpt-4'
    qa = ConversationalRetrievalChain.from_llm(model, retriever=retriever)

    return qa


def qa_loop(qa: Any):
    """
    Start a Question & Answer loop using the provided Q&A chain.

    Args:
        qa (ConversationalRetrievalChain): The Q&A chain for performing Q&A interactions.

    Returns:
        dict: A dictionary containing Q&A interactions and corresponding answers.
    """

    chat_history = []
    qa_log = {}
    print("Starting Q&A Prompt:")
    print(">> type `quit()` to exit")
    while True:
        question = input("Question:\n")
        if question == "quit()":
            break
        result = qa({"question": question, "chat_history": chat_history})
        chat_history.append((question, result["answer"]))
        print("Answer:\n")
        print(result["answer"])
        print("=" * 10)
        qa_log[question] = result
    return qa_log


def main(repo_url: str, user_name: str, root_dir: str, model_name: str):
    """
    Main function for executing the Q&A process.

    Args:
        repo_url (str): The URL of the repository to clone.
        root_dir (str): The directory where the repository will be cloned.
        model_name (str): The name of the model to be used for Q&A.

    Returns:
        None
    """
    # Clone the repository
    root_dir = clone_repo(repo_url, root_dir)

    # Load API keys
    load_creds()

    # Load documents from the repository
    docs = load_docs(root_dir)

    # Split documents into text chunks
    texts = text_splitter(docs)

    # Build a vector store and upload texts
    db = build_db(texts, user_name, f"test-open-ai-{int(time()) % 1000}")

    # Build the Q&A chain
    qa = build_qa_chain(db, model_name)

    # Start the Q&A loop
    qa_loop(qa)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    try:
        parser.add_argument(
            "--url",
            type=str,
            help="Repo url to clone.",
        )
        parser.add_argument(
            "--uname",
            type=str,
            help="Username for deeplake.",
        )
        parser.add_argument(
            "--model_name",
            type=str,
            default="gpt-3.5-turbo",
            help="model to use for Q&A.",
        )
        parser.add_argument(
            "--root_dir",
            type=str,
            default=None,
            help="location to store repo.",
        )
    except argparse.ArgumentError:
        logger.error("Argument Error Occured.")
        logger.error(parser.print_help())
        parser.print_help()

    args = parser.parse_args()
    root_dir = args.root_dir
    uname = args.uname
    model_name = args.model_name
    repo_url = args.url
    try:
        main(repo_url, uname, root_dir, model_name)
    finally:
        shutil.rmtree(root_dir)
