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

---
Prompt example
The following example demonstrates the creation of a prompt template, pass input variables, and templates as arguments.
from langchain import PromptTemplate
---

# Create a prompt template that has multiple input variables
multi_var_prompt = PromptTemplate(
     input_variables=["customerName", "feedbackFromCustomer"],
     template="""
     Human: Create an email to {customerName} in response to the following customer service feedback that was received from the customer: 
     <customer_feedback> 
          {feedbackFromCustomer}
     </customer_feedback>
     Assistant:"""
)
# Pass in values to the input variables
prompt = multi_var_prompt.format(customerName="John Doe",
          feedbackFromCustomer="""Hello AnyCompany, 
     I am very pleased with the recent experience I had when I called your customer support.
      I got an immediate call back, and the representative was very knowledgeable in fixing the problem. 
     We are very happy with the response provided and will consider recommending it to other businesses.
     """
)

---

Agent example
The following example demonstrates how to initialize an Agent, Tool, and LLM to form a chain and have the ZERO_SHOT ReAct agent call the in-built tool LLMMathChain to do math calculations separately and pass the result to the LLM for the final response.
---
from langchain.agents import load_tools
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain import LLMMathChain
from langchain_aws import ChatBedrock
from langchain.agents import AgentExecutor, create_react_agent

chat = ChatBedrock(model_id="anthropic.claude-3-sonnet-20240229-v1:0", model_kwargs={"temperature":0.1})

prompt_template = """Answer the following questions as best you can.
You have access to the following tools:\n\n{tools}\n\n
Use the following format:\n\nQuestion: the input question you must answer\n
Thought: you should always think about what to do\n
Action: the action to take, should be one of [{tool_names}]\n
Action Input: the input to the action\nObservation: the result of the action\n...
(this Thought/Action/Action Input/Observation can repeat N times)\n
Thought: I now know the final answer\n
Final Answer: the final answer to the original input question\n\nBegin!\n\n
Question: {input}\nThought:{agent_scratchpad}
"""
modelId = "anthropic.claude-3-sonnet-20240229-v1:0"

react_agent_llm = ChatBedrock(model_id=modelId, client=bedrock_client)
math_chain_llm = ChatBedrock(model_id=modelId, client=bedrock_client)

tools = load_tools([], llm=react_agent_llm)

llm_math_chain = LLMMathChain.from_llm(llm=math_chain_llm, verbose=True)

llm_math_chain.llm_chain.prompt.template = """Human: Given a question with a math problem, provide only a single line mathematical expression that solves the problem in the following format. Don't solve the expression only create a parsable expression.
```text
{{single line mathematical expression that solves the problem}}
```

Assistant:
Here is an example response with a single line mathematical expression for solving a math problem:
```text
37593**(1/5)
```

Human: {question}
Assistant:"""

tools.append(
    Tool.from_function(
         func=llm_math_chain.(opens in a new tab)run(opens in a new tab),
         name="Calculator",
         description="Useful for when you need to answer questions about math.",
    )
)

react_agent = create_react_agent(react_agent_llm,
    tools,
    PromptTemplate.from_template(prompt_template)
         # max_iteration=2,
         # return_intermediate_steps=True,
         # handle_parsing_errors=True,
    )

agent_executor = AgentExecutor(
agent=react_agent,
tools=tools,
verbose=True,
handle_parsing_errors=True,
max_iterations = 10 # useful when agent is stuck in a loop
)

agent_executor.invoke({"input": "What is the distance between San Francisco and Los Angeles? If I travel from San Francisco to Los Angeles with the speed of 40MPH how long will it take to reach?"})
