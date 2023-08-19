# Figaro
  An autonomous assistant for your daily life.
---

### Autonomous Agent Managing your life 
- [OpenAI model](https://platform.openai.com/docs/models) as Large Language model
- [LangChain](https://python.langchain.com/en/latest/modules/models/llms/integrations/huggingface_hub.html) as a Framework for LLM
- [Chainlit](https://docs.chainlit.io/langchain) for deploying.

## System Requirements

Tested with with Python 3.11 

---

## Install  and Run

1. Git clone the project 
2. Rename example.env to .env with `cp example.env .env`and input the OpenAI API key as follows. Get OpenAI API key from this [URL](https://platform.openai.com/account/api-keys). You need to create an account in OpenAI webiste if you haven't already.
   ```
   OPENAI_API_KEY=your_openai_api_key
   ```

3. Inside the project folder Create a virtualenv and activate it
   ```
   python3 -m venv .venv && source .venv/bin/activate
   ```

   If you also use conda to create the virtual environment.
   ```
   conda create -n .venv python=3.11 -y && source activate .venv
   ```

3. Install dependencies 
   ```
   # Python dependencies
   pip install -r requirements.txt   
   # Nodejs dependencies
   pnpm install
   ```
4. Authorize the email worker to subscribe to your emails 
   [TODO]

5. Run the main agent and its side worker 
   ```
   node workers/email-worker.js
   chainlit run app.py -w
   ```