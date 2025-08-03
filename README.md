# GenerativeAI_AWS_Bedrock #

The following example demonstrates how to create an instance of the Amazon Bedrock class and invoke an Amazon Titan LLM from the LangChain LLM module. The model_id is the value of the selected model available in Amazon Bedrock.
---
import boto3
from langchain_aws import BedrockLLM
bedrock_client = boto3.client('bedrock-runtime',region_name="us-east-1")
inference_modifiers = {"temperature": 0.3, "maxTokenCount": 512}
llm = BedrockLLM(
    client = bedrock_client,
    model_id="amazon.titan-tg1-large",
    model_kwargs =inference_modifiers
    streaming=True,
)
response = llm.invoke("What is the largest city in Vermont?")
print(response)

---

Custom models 
-
Amazon Bedrock can be used to customize FMs in order to improve their performance and create a better customer experience for specific use-case.  You can do this with FM continued pre-training or fine-tuning.
-
The following example demonstrates how you can get a response from an Amazon Bedrock custom model by passing a user request to the LLM. 
-
---

import boto3
from langchain_aws import BedrockLLM
custom_llm = BedrockLLM(
  credentials_profile_name = "bedrock-admin",
  provider = "cohere",
  model_id="<Custom model ARN>",  # ARN like 'arn:aws:bedrock:..' obtained from provisioned custom model
  model_kwargs ={"temperature": 1},
  streaming=True,
)
response = custom_llm.invoke("What is the recipe for mayonnaise?")
print(response)

---
Chat models example
The following example demonstrates how you can get a response from an LLM by passing a user request to the LLM.
-
Input
---
from langchain_aws import ChatBedrock as Bedrock
from langchain.schema import HumanMessage
chat = Bedrock(model_id="anthropic.claude-3-sonnet-20240229-v1:0", model_kwargs={"temperature":0.1})

messages = [
     HumanMessage(
          content="I would like to try Indian food, what do you suggest should I try?"
     )
]
chat.invoke(messages)

---
Embedding example 
The following example demonstrates how to call a BedrockEmbeddings client to send text to the Amazon Titan Embeddings model to get embeddings as a response.
---
from langchain_community.embeddings import BedrockEmbeddings

embeddings = BedrockEmbeddings(
    region_name="us-east-1",
    model_id="amazon.titan-embed-text-v1"
)

embeddings.embed_query("Cooper is a puppy that likes to eat beef")

