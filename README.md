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
