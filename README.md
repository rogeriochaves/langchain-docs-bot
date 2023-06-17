# Docs Bot

![Chatbot](https://github.com/rogeriochaves/langchain-docs-bot/assets/792201/a1a5b8d2-2963-4caa-8c65-86c63d67f0cb)

## Getting Started

To run the docs bot, clone this repo and install the dependencies

```
pip install -r requirements.txt
```

Then, create a `.env` file with your OpenAI key to use GPT APIs:

```
OPENAI_API_KEY=<your key here>
```

Now, run `update_db.py` file to generate the embeddings and index the docs. It will index everything under `docs/` folder, so you can replace that with your own docs if you want, and then:

```
python update_db.py
```

You are now ready to use the chatbot and search your docs:

```
chainlit run main.py -w
```

## Using a local LLM

In case you don't want to send your data over to OpenAI, you can try to use a local LLM, running even on your CPU with GPT4All. However, in my tests, they had a very bad performance, hardly passing the first chain step. If you want to try anyway, uncomment any of the `GPT4All` lines in `main.py`. Also, you will need to download the models from [gpt4all.io](https://gpt4all.io/) (for example the ggml-mpt-7b-instruct one), and save them under the `models/` folder.
