from dotenv import load_dotenv

load_dotenv()

import langchain
import langchain.schema
from langchain.chains.question_answering import load_qa_chain

from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain, SimpleSequentialChain
from update_db import persist_directory, embedding
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate
from langchain.chains.router import MultiPromptChain, MultiRouteChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain import HuggingFacePipeline

langchain.debug = True

vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

llm = ChatOpenAI(client=None, model="gpt-3.5-turbo", temperature=0) #, max_tokens=3000)
# GPT4all
# llm = HuggingFacePipeline.from_model_id(model_id="mosaicml/mpt-7b-instruct", task="text-generation")

# chat = ChatOpenAI(
#     client=None,
#     streaming=True,
#     callbacks=[StreamingStdOutCallbackHandler()],
#     temperature=0,
# )

qa = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectordb.as_retriever(),
    reduce_k_below_max_tokens=True
)

default_template = """You are a very basic chatbot just for triaging, you only engage in small talk, very shallow, like a "hello", "how is it going", but not much more than that.
For anything else, you say you don't have this information and you cannot answer, tell the user you can only answer questions about the company technical docs, this will redirect them to the right place

{history}
Human: {question}
Assistant:"""

memory = ConversationBufferWindowMemory(k=30)

# seq = SimpleSequentialChain() # type: ignore

destination_chains = {"DOCS": qa}

default_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["history", "question"], template=default_template
    ),
    memory=memory,
    output_key="answer",
)


MULTI_PROMPT_ROUTER_TEMPLATE = """You help triaging user requests. Given a raw text input, output a JSON that classifies the destination as:

DOCS: if user is asking a seemingly technical question, programming questions or company-specific questions
DEFAULT: if user is just chit-chatting or basic knowledge questions

<< JSON FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{{
    "destination": string // either "DOCS" or "DEFAULT"
    "next_inputs": string // the original input
}}}}
```

<< INPUT >>
{{input}}

<< OUTPUT (remember to include the ```json)>>"""

router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format()
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(next_inputs_inner_key="question"),
)

router_chain = LLMRouterChain.from_llm(llm, router_prompt)


# prompt = PromptTemplate(
#     input_variables=["history", "human_input"],
#     template=template
# )

# chatbot_chain = LLMChain(
#     llm=llm,
#     prompt=prompt,
#     memory=ConversationBufferWindowMemory(k=30),
# )


class MixedMultiRouteChain(MultiRouteChain):
    @property
    def output_keys(self):
        return ["answer"]


chain = MixedMultiRouteChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=default_chain,
    verbose=True,
)

while True:
    print("> ", end="")
    user_input = input()
    attempts = 0
    # llm.temperature = 0
    while True:
        try:
            output = chain.run(user_input)
            break
        except Exception as e:
            attempts += 1
            # llm.temperature += 0.1
            if attempts >= 1:
                raise e
    print(output)

# qa.run("What is langchain?")
# qa.run("And how much is 2 + 2?")
