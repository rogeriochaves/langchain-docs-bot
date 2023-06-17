import re
from typing import Any, Dict, Literal, TypedDict, cast
from dotenv import load_dotenv

load_dotenv()

import langchain
import langchain.schema

from langchain.chat_models import ChatOpenAI
from langchain.llms import GPT4All
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA
from update_db import persist_directory, embedding
from langchain import (
    LLMChain,
    PromptTemplate,
)
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseOutputParser, OutputParserException
from langchain.callbacks.manager import Callbacks
import chainlit as cl

langchain.debug = True

vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

llm = ChatOpenAI(client=None, model="gpt-3.5-turbo-16k", temperature=0, streaming=True)
# llm = GPT4All(model="./models/ggml-mpt-7b-instruct.bin", backend="mpt", verbose=True, temp=0.1, repeat_penalty=2)  # type: ignore
# llm = GPT4All(model="./models/ggml-gpt4all-j-v1.3-groovy.bin", backend="gptj", verbose=True, temp=0)  # type: ignore

memory = ConversationBufferWindowMemory(k=30, memory_key="history")


class RoutingChainOutput(TypedDict):
    action: Literal["SEARCH", "REPLY"]
    param: str


def simple_key_extract(key: str, output: str) -> str:
    found = re.search(f"{key}\\s?: (.*)", output, flags=re.IGNORECASE)
    if found is None:
        raise OutputParserException(f"Key '{key}:' not found on {output}")

    return found[1]


class RoutingParser(BaseOutputParser[RoutingChainOutput]):
    def parse(self, output: str) -> Dict[str, Any]:
        return {
            "action": cast(
                Literal["SEARCH", "REPLY"], simple_key_extract("Action", output)
            ),
            "param": simple_key_extract("Param", output),
        }


class RoutingChain(LLMChain):
    pass


routing_chain = RoutingChain(
    llm=llm,
    memory=memory,
    prompt=PromptTemplate(
        template="""
        You are a chatbot that primarly helps users to search on docs, but you may also chit-chat with them.
        Given a user input, choose the correction Action:

        SEARCH: if user is asking a question, it takes the search query as param
        REPLY: if user is just chit-chatting like greeting or asking how are you, it takes the reply from the chatbot as a param

        ====================

        Input: hello there
        Action: REPLY
        Param: hey there, what are you looking for?

        Input: how does langchain work?
        Action: SEARCH
        Param: langchain how it works

        Input: code example of vector db
        Action: SEARCH
        Param: vector db code example

        Input: how is it going?
        Action: REPLY
        Param: I'm going well, how about you?

        {history}
        Input: {input}
    """,
        input_variables=["history", "input"],
        output_parser=RoutingParser(),
    ),
)


qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectordb.as_retriever(),
)


def conversation(input: str, callbacks: Callbacks) -> str:
    route = cast(
        RoutingChainOutput,
        routing_chain.predict_and_parse(callbacks=callbacks, input=input),
    )
    if route["action"] == "REPLY":
        return route["param"]
    elif route["action"] == "SEARCH":
        result = qa_chain.run(route["param"], callbacks=callbacks)

        return result
    else:
        return f"unknown action {route['action']}"


@cl.langchain_factory(use_async=False)
def factory():
    return conversation
