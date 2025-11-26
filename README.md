# My Bedrock Agents Comparison
This is a sample demo on how to compare different model result that is using Bedrock Agent specifically InvokeAgent API.

To run the app. Go to your Bedrock Console and create/duplicate your agents. Each agent should correspond to different models. Get the Agent ID and Alias ID

Clone the repo and cd into the repo

Login into your AWS credentials. Example I'm using AWS Identity Center
`aws sso login --profile default`

Initiatize Virtual Environment with Python
`python3 -m venv venv`
`source venv/bin/activate`

Install the dependencies
`pip install -q -r requirements.txt`

Run the app with Streamlit
`streamlit run app.py`
