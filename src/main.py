"""Main python script."""

import subprocess
from typing import Tuple, Optional, Any, List
import os
import json
from tempfile import TemporaryDirectory
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from time import time
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import argparse
import logging
import shutil
import pdb

logger = logging.getLogger(name="QA_PROMPT")

def run_command(cmd: str) -> None:
    """Execute console commands."""

    print("="*10)
    print(f"Executing: {cmd}")
    subprocess.run(cmd)
    print("="*10)

def clone_repo(URL: str, root_dir: Optional[str]=None):
    """Clone repo"""
    if not root_dir:
        root_dir = TemporaryDirectory(prefix="qa_with_code").name
    
    run_command(f"git clone {URL} {root_dir}")

    return root_dir

def load_creds() -> None:
    """Load API Keys."""
    with open("creds.json", 'r') as f:
        keys = json.loads(f.read())
    os.environ["OPENAI_API_KEY"] = keys["open_ai"]
    os.environ["ACTIVELOOP_TOKEN"] = keys["activeloop_ai"]

def load_docs(root_dir: str) -> List[Any]:
    """Load docs."""
    docs = []
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            try:
                loader = TextLoader(os.path.join(dirpath, file), encoding="utf-8")
                docs.extend(loader.load_and_split())
            except Exception as e:
                pass
    return docs

def text_splitter(docs: List[Any], chunk_size: int=1000) -> Any:
    """Return text chunks as vector format."""
    splits = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = splits.split_documents(docs)
    return texts

def build_db(texts: Any, user_name: str, db_name: Optional[str]) -> Any:
    """Upload texts to db"""
    embeddings = OpenAIEmbeddings(disallowed_special=())
    db = DeepLake(
        dataset_path=f"hub://{user_name}/{db_name}",
        embedding_function=embeddings
        )
    db.add_documents(texts)
    return db

def build_qa_chain(db: Any, model_name: str) -> Any:
    """Build Q&A chain through prompts"""

    retriever = db.as_retriever()
    model = ChatOpenAI(model_name=model_name)  # switch to 'gpt-4'
    qa = ConversationalRetrievalChain.from_llm(model, retriever=retriever)

    return qa

def qa_loop(qa: Any):
    """Q&A Loop."""

    chat_history = []
    qa_log = {}
    print("Starting Q&A Prompt:")
    print(">> type `quit()` to exit")
    while(True):
        question = input("Question:\n")
        if question == "quit()":
            break
        result = qa({"question": question, "chat_history": chat_history})
        chat_history.append((question, result["answer"]))
        print(f"Answer:\n")
        print(result["answer"])
        print("="*10)
        qa_log[question] = result
    return qa_log


def main(repo_url: str, root_dir: str, model_name: str):
    """Main function."""

    # Clone repo.
    root_dir = clone_repo(repo_url, root_dir)
    load_creds()
    docs = load_docs(root_dir)
    texts = text_splitter(docs)
    db = build_db(texts, "agniakash25", f"test-open-ai-{int(time())%1000}")
    qa = build_qa_chain(db, model_name)
    qa_log = qa_loop(qa)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    try:
        parser.add_argument(
            "--root_dir",
            type=str,
            default=None,
            help="location to store repo.",
        )
        parser.add_argument(
            "--url",
            type=str,
            help="Repo url to clone.",
        )
        parser.add_argument(
            "--model_name",
            type=str,
            default="gpt-3.5-turbo",
            help="model to use for Q&A.",
        )
    except argparse.ArgumentError:
        logger.error("Argument Error Occured.")
        logger.error(parser.print_help())
        parser.print_help()
    
    args = parser.parse_args()
    root_dir = args.root_dir
    model_name = args.model_name
    url = args.url
    try:
        main(url, root_dir, model_name)
    except:
        shutil.rmtree(root_dir)

