# RAG demo

This repo is intended to serve as a starting point for building and running a RAG + LLM chat application locally.

## Get started

Create an env and activate

```
conda create --name RAG-demo python=3.9.16
conda activate RAG-demo
```

Install packages.
llama.cpp configured for metal, modify pyproject.toml to suit your needs.

```
pip install -r requirements.txt
pip install llama-cpp-python
```

Supply `MODEL_PATH` which refers to quantized llm, see [https://github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) for instructions

```
cp .env_example .env
```

Place data in the data folder and run `gen_vector_db.py` to create db.

To start the chat client on localhost:8000

```
chainlit run app.py -w
```
