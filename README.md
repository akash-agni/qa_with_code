# Talk to your Code.

Welcome to the Question & Answer (Q&A) repository! This repository contains a Python script that utilizes conversational retrieval chains to perform Q&A interactions with a given set of documents. The script leverages the OpenAI models for answering questions based on the provided documents.

## Getting Started

To get started, follow these steps:

1. Clone this repository to your local machine using the following command

2. Navigate to the cloned directory

3. Install the required dependencies. It's recommended to use a virtual environment

4. Create a file named `creds.json` in the repository root and add your OpenAI and ActiveLoop API keys:
```json
{
    "open_ai": "OPENAI_API_KEY",
    "activeloop_ai": "ACTIVELOOP_TOKEN"
}
```

5. Run the Q&A script with the desired repository URL, root directory, and model name
```
python main.py --url <repository_url> --uname <user_name> --root_dir <desired_root_dir> --model_name <desired_model_name>
```

## Script Details

The main script (main.py) performs the following steps:

<ol>
    <li>Clones a repository from the specified URL into a given root directory.</li>
    <li>Loads API keys from creds.json for OpenAI and ActiveLoop.</li>
    <li>Loads documents from the cloned repository.</li>
    <li>Splits documents into text chunks and builds a vector store.</li>
    <li>Builds a Q&A chain using the provided model name.</li>
    <li>Initiates a Q&A loop, interacting with the user through the console.</li>
</ol>

## Note
Make sure to replace <repository_url>, <user_name> <desired_root_dir>, and <desired_model_name> with actual values when using the script. <desired_root_dir> and <desired_model_name> are optional by default temporary directories and `gpt-3.5-turbo` model would be used.

> Disclaimer: This repository is provided as-is and is not affiliated with OpenAI or ActiveLoop.