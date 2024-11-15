# chatbot.py

from abc import ABC, abstractmethod

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # 导入提示模板相关类
from langchain_core.messages import HumanMessage  # 导入消息类
from langchain_core.runnables.history import RunnableWithMessageHistory  # 导入带有消息历史的可运行类
from langchain.graphs import NetworkxEntityGraph  # 导入 LangGraph 反思机制
import openai
import os
from logger import LOG  # 导入日志工具
from chat_history import get_session_history


class ChatBot(ABC):
    """
    聊天机器人基类，提供聊天功能。
    """
    def __init__(self, prompt_file="./prompts/chatbot.txt", session_id=None):
        self.prompt_file = prompt_file
        self.session_id = session_id if session_id else "default_session_id"
        self.prompt = self.load_prompt()
        # LOG.debug(f"[ChatBot Prompt]{self.prompt}")
        self.create_chatbot()

    def load_prompt(self):
        """
        从文件加载系统提示语。
        """
        try:
            with open(self.prompt_file, "r", encoding="utf-8") as file:
                return file.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到提示文件 {self.prompt_file}!")


    def create_chatbot(self):
        """
        初始化聊天机器人，包括系统提示和消息历史记录。
        """
        # 创建聊天提示模板，包括系统提示和消息占位符
        system_prompt = ChatPromptTemplate.from_messages([
            ("system", self.prompt),  # 系统提示部分
            MessagesPlaceholder(variable_name="messages"),  # 消息占位符
        ])

        openai.api_key = os.getenv("OPENAI_API_KEY")
        # 初始化 ChatOllama 模型，配置参数
        self.chatbot = system_prompt | ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.5,
            max_tokens=4096
        )

        # 将聊天机器人与消息历史记录关联
        self.chatbot_with_history = RunnableWithMessageHistory(self.chatbot, get_session_history)
        
        # 初始化图数据库（NetworkxEntityGraph）
        self.graph = NetworkxEntityGraph()


    def chat_with_history(self, user_input, session_id=None):
        """
        处理用户输入，生成包含聊天历史的回复。

        参数:
            user_input (str): 用户输入的消息
            session_id (str, optional): 会话的唯一标识符

        返回:
            str: AI 生成的回复
        """
        if session_id is None:
            session_id = self.session_id
    
        # 生成初步回复
        initial_response = self.chatbot_with_history.invoke(
            [HumanMessage(content=user_input)],  # 将用户输入封装为 HumanMessage
            {"configurable": {"session_id": session_id}},  # 传入配置，包括会话ID
        )
        
        LOG.debug(f"[ChatBot] 初始回应: {initial_response.content}")  # 记录调试日志
        
        # 将用户输入和初步回复作为图节点存储
        self.graph.add_node(user_input)
        self.graph.add_node(initial_response.content)
        self.graph.add_edge(user_input, initial_response.content)  # 添加连接
        
        # 使用 LangGraph 进行反思，提升生成内容的质量
        improved_response = self.reflection_rounds(initial_response.content)

        LOG.debug(f"[ChatBot] 改进后的回应: {improved_response}")  # 记录调试日志
        
        # 最终回应会经过 LangGraph 反思机制的优化
        return improved_response  # 返回生成的最终回应内容
    
    def reflection_rounds(self, initial_response, rounds=3):
        """
        通过多轮反思提升回应的质量。

        参数:
            initial_response (str): 初步生成的回应
            rounds (int): 反思的轮数（默认3轮）

        返回:
            str: 经多轮反思后生成的最终回应
        """
        # 模拟反思机制（根据图的结构进行提升）
        for i in range(rounds):
            # 可以通过图结构来计算与当前回应相关的节点，基于节点的连接性提升回应质量
            LOG.debug(f"[反思轮 {i+1}] 当前回应: {initial_response}")
            # 例如，基于图中的相似节点或权重来增强回应内容
            # 这里我们可以简单地进行文本增强，您可以根据自己的需求修改
            initial_response = f"Enhanced: {initial_response}"

        return initial_response