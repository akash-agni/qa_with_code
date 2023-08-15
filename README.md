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

## Example

```
Starting Q&A Prompt:
>> type `quit()` to exit
Question:
What are Hooks ?
Answer:

Hooks are a mechanism in Kedro that allow you to add extra behavior to Kedro's main execution. They provide a way to inject additional code at specific points in the execution of your Kedro pipeline. This can be useful for tasks such as logging, data validation, or tracking metrics. Hooks consist of a specification and an implementation and can be registered in your project's settings file.
==========
Question:
List all the hooks that developers can use to customize their pipelines.
Answer:

Developers can use the following hooks to customize their pipelines:

- `before_node_run`: This hook is executed before each node in the pipeline is run.
- `after_node_run`: This hook is executed after each node in the pipeline is run.
- `before_pipeline_run`: This hook is executed before the pipeline is run.
- `after_pipeline_run`: This hook is executed after the pipeline is run.
- `before_dataset_loaded`: This hook is executed before a dataset is loaded.
- `after_dataset_loaded`: This hook is executed after a dataset is loaded.
- `before_dataset_saved`: This hook is executed before a dataset is saved.
- `after_dataset_saved`: This hook is executed after a dataset is saved.
- `after_context_created`: This hook is executed after the Kedro context is created.

By using these hooks, developers can add custom behavior or modify the execution flow of their pipelines.
==========
Question:
create a custom hook to measure the amount of time each node takes to execute and print the result.
Answer:

Developers can create a custom hook to measure the amount of time each node takes to execute and print the result by following these steps:        

1. Define a class for your custom hook and import the necessary modules:

Starting Q&A Prompt:
>> type `quit()` to exit
Question:
What are Hooks ?
Answer:

Hooks are a mechanism in Kedro that allow you to add extra behavior to Kedro's main execution.
They provide a way to inject additional code at specific points in the execution of your Kedro pipeline.
This can be useful for tasks such as logging, data validation, or tracking metrics.
Hooks consist of a specification and an implementation and can be registered in your project's settings file.

==========

Question:
List all the hooks that developers can use to customize their pipelines.
Answer:

Developers can use the following hooks to customize their pipelines:

- `before_node_run`: This hook is executed before each node in the pipeline is run.
- `after_node_run`: This hook is executed after each node in the pipeline is run.
- `before_pipeline_run`: This hook is executed before the pipeline is run.
- `after_pipeline_run`: This hook is executed after the pipeline is run.
- `before_dataset_loaded`: This hook is executed before a dataset is loaded.
- `after_dataset_loaded`: This hook is executed after a dataset is loaded.
- `before_dataset_saved`: This hook is executed before a dataset is saved.
- `after_dataset_saved`: This hook is executed after a dataset is saved.
- `after_context_created`: This hook is executed after the Kedro context is created.

By using these hooks, developers can add custom behavior or modify the execution flow of their pipelines.

==========

Question:
create a custom hook to measure the amount of time each node takes to execute and print the result.
Answer:

Developers can create a custom hook to measure the amount of time each node takes to execute and print the result by following these steps:        

1. Define a class for your custom hook and import the necessary modules:

\```python
import time
from kedro.framework.hooks import hook_impl
from kedro.pipeline.node import Node
\```

2. Implement the `before_node_run` method in your hook class. This method will be called before each node is executed:

\```python
@hook_impl
def before_node_run(self, node: Node) -> None:
    self._start_time = time.time()
\```

3. Implement the `after_node_run` method in your hook class. This method will be called after each node is executed:

\```python
@hook_impl
def after_node_run(self, node: Node, inputs: Dict[str, Any]) -> None:
    execution_time = time.time() - self._start_time
    print(f"Node {node.name} took {execution_time} seconds to execute.")
\```

4. Register your custom hook in your project's `hooks.py` file:

\```python
from my_package.my_hooks import CustomTimingHook

hooks = [CustomTimingHook()]
\```

Now, when you run your Kedro pipeline, each node's execution time will be printed in the console.

==========

Question:
quit()

Ending conversational chain, thank you!!
```