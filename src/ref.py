import os
from typing import Annotated, List
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama.chat_models import ChatOllama
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict
from IPython.display import Markdown, display
import asyncio

def track_steps(func):
    step_counter = {'count': 0}

    def wrapper(event, *args, **kwargs):
        step_counter['count'] += 1
        #display(Markdown(f"## Round {step_counter['count']}"))
        print(f"## Round {step_counter['count']}")  
        return func(event, *args, **kwargs)

    return wrapper

# 定义状态模式类，用于StateGraph
class StateSchema(TypedDict):
    messages: Annotated[List, add_messages]

class ChatAssistant:
    MAX_ROUND = 4

    def __init__(self, writer_model_url: str, reflect_model_url: str, reflect_prompt: str, writer_prompt_file: str = "./prompts/chatbot.txt"):

        # 初始化写作和反思模型
        self.writer = self._initialize_writer(writer_model_url, writer_prompt_file)
        self.reflect = self._initialize_reflect(reflect_model_url, reflect_prompt)

        # 初始化状态图
        self.builder = self._initialize_state_graph()
        self.memory = MemorySaver()
        self.graph = self.builder.compile(checkpointer=self.memory)

    def _initialize_writer(self, base_url: str, prompt_file: str):
        
        with open(prompt_file, 'r', encoding="utf-8") as f:
            writer_prompt = f.read();
            
        writer_prompt = ChatPromptTemplate.from_messages([
            ("system", writer_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])
        return writer_prompt | ChatOllama(
            model="qwen2.5:3b", max_tokens=8192, temperature=1.2, base_url=base_url
        )

    def _initialize_reflect(self, base_url: str, prompt: str):
        reflection_prompt = ChatPromptTemplate.from_messages([
            ("system", prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])
        return reflection_prompt | ChatOllama(
            model="qwen2.5:3b", max_tokens=8192, temperature=0.2, base_url=base_url
        )

    async def generation_node(self, state: StateSchema) -> StateSchema:
        return {"messages": [await self.writer.ainvoke(state['messages'])]}

    async def reflection_node(self, state: StateSchema) -> StateSchema:
        cls_map = {"ai": HumanMessage, "human": AIMessage}
        translated = [state['messages'][0]] + [
            cls_map[msg.type](content=msg.content) for msg in state['messages'][1:]
        ]
        res = await self.reflect.ainvoke(translated)
        return {"messages": [HumanMessage(content=res.content)]}

    def should_continue(self, state: StateSchema):
        return END if len(state["messages"]) > self.MAX_ROUND else "reflect"

    def _initialize_state_graph(self):
        builder = StateGraph(StateSchema)  # 使用StateSchema而不是字典
        builder.add_node("writer", self.generation_node)
        builder.add_node("reflect", self.reflection_node)
        builder.add_edge(START, "writer")
        builder.add_conditional_edges("writer", self.should_continue)
        builder.add_edge("reflect", "writer")
        return builder

    @track_steps
    def pretty_print_event_markdown(self, event):
        if 'writer' in event:
            generate_md = "#### 内容生成:\n" + "".join(f"- {msg.content}\n" for msg in event['writer']['messages'])
            #display(Markdown(generate_md))
            print(generate_md)  # 修改为 print
        if 'reflect' in event:
            reflect_md = "#### 反思:\n" + "".join(f"- {msg.content}\n" for msg in event['reflect']['messages'])
            #display(Markdown(reflect_md))
            print(reflect_md)  # 修改为 print

    async def start_chat(self, input_text: str):
        inputs = {"messages": [HumanMessage(content=input_text)]}
        config = {"configurable": {"thread_id": "1"}}
        
        final_reflection = None
        
        async for event in self.graph.astream(inputs, config):
            self.pretty_print_event_markdown(event)
            
            # 判断是否为反思节点
            if 'writer' in event:
                final_reflection = event['writer']['messages'][0].content
        
        # 返回最终的反思结果
        return final_reflection

# 程序入口
if __name__ == "__main__":
    # 使用示例
    reflect_promte = """###角色: 你是一位严格且公正的内容评审员，专注于评估主聊天机器人生成的内容。你的职责是确保主聊天机器人输出的内容符合其提示词的标准，具备结构化、逻辑性、内容深度和适合PowerPoint演示的特点。

目标: 反思体在主聊天机器人的输出基础上进行深入评估，确认内容是否符合主聊天机器人的提示词要求，并提出优化建议，以确保内容适合演示文稿
    """
    assistant = ChatAssistant(writer_model_url="http://172.16.3.199:11434", reflect_model_url="http://172.16.3.199:11434", reflect_prompt=reflect_promte)
    
    # 使用 asyncio.run() 来运行异步函数
    final_reflection = asyncio.run(assistant.start_chat("项目管理与AI大模型"))
    
    print("\n\n final_reflection: \n\n" + final_reflection)