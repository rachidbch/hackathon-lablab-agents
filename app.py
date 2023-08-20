from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import os
import chainlit as cl
from chainlit import user_session

from fastapi import Request
# from fastapi import FastAPI

from chainlit.server import app
# import json

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# The embeddings we'll use
embeddings = OpenAIEmbeddings()

# [TODO] Push this into its own module
# Global Context Hack!
# We can't access the user session from the API endpoint. Thus we can't create message from there neither.
# As a (horrible!) work around, we pre-create a pool of message attached to each user session at each chat start
# A pool is mapped to its session_id in the global context so we can reach to it from the API endpoint handler.
# We checked that messages pre-created with a session can be modified and sent from an endpoint handler.


class GlobalContext:
    def __init__(self):
        self.user_sessions = {}
        self.users = {}

    # Get a sid (sid) from a uid
    # [TODO] Rename sid? 
    def user(self, uid=None):
        # Simple helper so that `gctx.user()`can get the global context user session id if current session is defined 
        if not uid:
            if user_session:                       # user_session is only defined when the whole chainlit session context is loaded 
                return user_session.get("id")      # In that case, we can easily access the sid

        # Most of the time we want a handle on the global context session of a user by its id 
        if uid not in self.users:
            return None 

        return self.users[uid]

    def set_user(self, uid, sid=None):
        if not sid:
            if user_session:
                sid = user_session.get('id') 
            else:
                # [TODO] Log error
                pass

        self.users.update({ uid: sid })  

    # Get a user_session from a sid 
    def user_session(self, sid=None):
        # Simple helper so that `gctx.user_session()`can get the global context user session if current session is defined 
        if not sid:
            if user_session:
                sid = user_session.get("id")
            else:
                return None

        # Most of the time we want a handle on the global context session of a user by its id 
        if sid not in self.user_sessions:
            self.user_sessions[sid] = {}
        return UserSession(self.user_sessions[sid])

class UserSession:
    def __init__(self, session_dict):
        self.session_dict = session_dict

    def set(self, key, value):
        self.session_dict[key] = value

    def get(self, key):
        return self.session_dict.get(key)

gctx = GlobalContext()

# Long texts needs to be splited into chunks that are smaller than our LLM context
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# Our LLM support a system prompt and a user prompt 
system_template = """You are Figaro a virtual assistant, that always tries to be helpful and friendly.

    a) If the user engage normal conversation, just keep the conversation nicely going. 
    b) If the user asks a question, use the following pieces of context to answer the users question.
       If you thing it is pertinent, return a "SOURCES" part in your answer (
       (In that case The "SOURCES" part should be a reference to the source of the document from which you got your answer)

Examples of your responses should be:

    - Possible Conversation Response:
```
The answer is foo
SOURCES: xyz
```
    - Possible Question Response:
```
I'm fine and you :)
```

Begin!
----------------
{summaries}"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}


# When a user is identified, we prepopulate it's gobal context session with pre-maid messages 
# See Global Context Hack above
def init_global_context():

    global gctx 
    sid = gctx.user() 

    msgs_pool= []
    for i in range(10):
        pre_created_msg = cl.Message(content="Email received")
        msgs_pool.append(pre_created_msg)
    gctx.user_session(sid).set('pool', msgs_pool)


@cl.on_chat_start
async def on_chat_start():
    
    global gctx

    # Hack! No Session context Workaround 
    init_global_context()

    # [TEST] Map a test user to current sid 
    gctx.set_user('testuserhackathon123@gmail.com')

    # Greeting new user
    elements = [
        cl.Image(name="image1", display="inline", path="./Figaro.png")
    ]
    await cl.Message(content="Hey there! I'm Figaro. How can I help?", elements=elements).send()

    # Setup a langchain chain for this user 
    # We use a Chroma vector store 
    vectordb = Chroma("Figaro", embeddings)
    
    # Langchain requires at list one document for this typo of chain
    metadatas = [{"type": 'session', "source": 'system'}]
    await cl.make_async(vectordb.add_texts)(['Figaro is the best'], ids=['0'], metadatas=metadatas)

    # We want our searches to be sourced 
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        ChatOpenAI(temperature=0, streaming=True),
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
    )


    gctx.user_session().set("vectordb", vectordb)
    gctx.user_session().set("chain", chain)


@cl.on_message
async def on_message(message):

    chain = gctx.user_session().get("chain")  # type: RetrievalQAWithSourcesChain
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )

    cb.answer_reached = True                          # Normal behaviour is to stream only the final answer (in the case of a CoT for instance) 
                                                      # Here we want to stream everything. cb.answer_reached set to True forces that beahviour.
                                                      # [TODO] Does this stream really everything to the user or is Chainlit doing its own fitering? 
    res = await chain.acall(message, callbacks=[cb])

    # We got our answer
    # It's important to understand that at this point, because we passed the Chainlit AsyncLangchainCallBackHandler, 
    # the answer has already been streamed to the UI.
    answer = res["answer"]
    
    # We parse the sources ids from the answer and reconsitute sources metadata from the sources stored in the Session context 
    sources = res["sources"].strip()
    source_elements = []

    # Get the metadata and texts from the user session
    metadatas = gctx.user_session().get("metadatas")
    if metadatas:
        all_sources = [m["source"] for m in metadatas]
    else:
        all_sources = None
    texts = gctx.user_session().get("texts")

    if sources:
        found_sources = []
        # Add the sources to the message
        for source in sources.split(","):
            source_name = source.strip().replace(".", "")
            # Get the index of the source
            try:
                index = all_sources.index(source_name)
            except ValueError:
                continue
            text = texts[index]
            found_sources.append(source_name)
            # Create the text element referenced in the message
            source_elements.append(cl.Text(content=text, name=source_name))

        if found_sources:
            answer += f"\nSources: {', '.join(found_sources)}"
        else:
            answer += "\nNo sources found"

    # Beware everything is asynchrounous here.
    # We may reach this point BEFORE or AFTER Chainlit has streamed the final answer to the Web UI
    # If we're late, we must update the final message sent to the UI 
    if cb.has_streamed_final_answer:
        cb.final_stream.elements = source_elements
        await cb.final_stream.update()
    # Other wise we send the all answer and sources  ourselves
    # [TODO] No risk to have the answer sent twice; once by ourself and once by Chainlit afterwards?
    else:
        await cl.Message(content=answer, elements=source_elements).send()

@app.post("/event")
async def event(request: Request):
    global gctx 

    data = await request.json()

    # Parse the email data
    email = parse_email(data)  

    if email:
        print(f"\nEmail received:\n {email}")
        id = email.get('id')
        email_id = email.get('email_id')
        subject = subject = email.get('subject')
        sender = email.get('from')
        contact = email.get('contact')
        recipients = email.get('recipients')
        sent_at = email.get('sent_at')
        received_at = email.get('received_at')
        body = email.get('body')
        
        if (user_email := recipients[0]):

            # Send a notification to the web chat user
            sid = gctx.user(user_email)
            notification_msg = gctx.user_session(sid).get('pool').pop()
            await notification_msg.send()

            # Decide if the email is about a meeting
            chain = gctx.user_session(sid).get('chain')

            message = f"""Is this email discussing a meeting? Answer by yes or no.
            email:
            {email}
            """
            res = await chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])

            print(f"""Is this email related to a meeting: 
            {res}
            """)
            # Embed the new email
            vectordb = gctx.user_session(sid).get('vectordb')
            index_email([email], vectordb)



    return {"status": "OK"}

#@app.post("/event")
#async def event(request: Request):
#    global gctx 
#
#    data = await request.json()
#
#    sid = gctx.user('testuserhackathon123@gmail.com')
#    notification_msg = gctx.user_session(sid).get('pool').pop()
#    # TODO moodify content and time stamp of the message
#    await notification_msg.send()
#
#    return {"status": "OK"}


# [TODO] Push into utils directory
def parse_email(email_dict):
    
    clean_dict = {}
    
    clean_dict['id'] = email_dict.get('id', '')

    event = email_dict.get('event', {})
    clean_dict['email_id'] = event.get('id', '')
    clean_dict['body'] = event.get('decodedContent', '')

    headers = event['payload'].get('headers', [])

    for header in headers:
        if header['name'] == 'From':
            clean_dict['contact'] = header['value'].split('<')[0].strip() if '<' in header['value'] else header['value']
            clean_dict['from'] = header['value'].split('<')[1].replace('>', '').strip() if '<' in header['value'] else header['value']
        elif header['name'] == 'To':
            clean_dict['recipients'] = [recipient.strip() for recipient in header['value'].split(',')]
        elif header['name'] == 'Date':
            clean_dict['sent_at'] = header['value']
        elif header['name'] == 'Subject':
            clean_dict['subject'] = header['value']

    clean_dict['received_at'] = email_dict.get('indexed_at_ms', '')

    return clean_dict


# Ingest emails into the vector database
async def index_emails(emails, vectordb):

    # Generate unique ids for the documents
    # doc_ids = [str(uuid.uuid1()) for _ in range(len(docs))]
    docs = emails
    doc_ids = [str(email.id) for email in emails]

    # [TODO] We may need to split the emails as they could be longer than the context

    # Check if documents exist in the database
    new_docs = []
    new_doc_ids = []
    for doc, doc_id in zip(docs, doc_ids):
        if doc_id not in vectordb:
            new_docs.append(doc)
            new_doc_ids.append(doc_id)
            #print(f"Document '{doc}' added to the database with id '{doc_id}'.")
        else:
            #print(f"Document '{doc}' already exists in the database with id '{doc_id}'.")
            pass

    # Add new documents to the database along with their metadata
    if len(new_docs) > 0:
        metadatas = [{"type": "email", "source": email.id} for email in new_docs]  
        await cl.make_async(vectordb.add_documents)(new_docs, embeddings, ids=new_doc_ids, metadatas=metadatas)

# Enable uploading a document to the vector database
#  -  Inactive for now 
#  - [TODO] Add as a seperate workflow
async def ingest_user_doc_workflow():
    files = None
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a text file to begin!",
            accept=["text/plain"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]

    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    # Decode the file
    text = file.content.decode("utf-8")

    # Split the text into chunks
    texts = text_splitter.split_text(text)

    # Create a metadata for each chunk
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    docsearch = await cl.make_async(Chroma.from_texts)(
        texts, embeddings, metadatas=metadatas
    )

    # Create a chain that uses the Chroma vector store
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        ChatOpenAI(temperature=0, streaming=True),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
    )

    # Save the metadata and texts in the user session
    cl.user_session.set("metadatas", metadatas)
    cl.user_session.set("texts", texts)

    # Let the user know that the system is ready
    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)
