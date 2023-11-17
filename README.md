# RAG demo

This repo is intended to serve as a starting point for building and running a RAG + LLM chat application locally.

## Get started

Create an env and activate

```
conda create --name RAG-demo python=3.10
conda activate RAG-demo
```

Install packages.
llama.cpp configured for metal, modify requirements.txt `llama-cpp-python` section if you're not using a modern mac

```
pip install -r requirements.txt
```

Supply env var `MODEL_PATH` which refers to a path to quantized llm, see [https://github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) for instructions

```
cp .env_example .env
```

Place data in the data folder and run `gen_vector_db.py` to create db.

To start the chat client on localhost:8000

```
chainlit run app.py -w
```
