from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
import openai
import gradio as gr
import os
from pptx import Presentation
from pptx.util import Inches
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE
from io import BytesIO

from input_parser import parse_input_text
from ppt_generator import generate_presentation
from template_manager import load_template, get_layout_mapping, print_layouts
from layout_manager import LayoutManager
from config import Config
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.chains import LLMChain
from logger import LOG

# 设置 OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(temperature=1.0, model="gpt-3.5-turbo", max_tokens=1000)

config = Config()  # 加载配置文件
# 保存生成内容的全局变量
latest_content = {"text": ""}


def load_prompt_from_file(filepath: str) -> str:
    """从文件中读取提示词模板"""
    with open(filepath, "r", encoding="utf-8") as file:
        prompt_template = file.read()
    return prompt_template


# 定义 LLM 生成 Markdown 内容的函数
def generate_markdown_from_input(message, history) -> str:

    history_langchain_format = []
    for msg in history:
        if msg["role"] == "user":
            history_langchain_format.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history_langchain_format.append(AIMessage(content=msg["content"]))

    """根据用户输入和提示词模板生成 Markdown 内容"""
    # 翻译任务指令始终由 System 角色承担
    template = load_prompt_from_file("prompts/formatter.txt")
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # 待翻译文本由 Human 角色输入
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    # 使用 System 和 Human 角色的提示模板构造 ChatPromptTemplate
    chat_prompt_template = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    chain = LLMChain(llm=llm, prompt=chat_prompt_template, verbose=True)

    LOG.info(f"用户输入: {message}")

    result = chain.run({"text": message})  #

    latest_content["text"] = result

    return result


def generate_pptx_from_content():
    """基于最新聊天内容生成 PowerPoint 幻灯片"""
    input_text = latest_content["text"]

    LOG.info(f"最新天内容: {input_text}")

    prs = load_template(config.ppt_template)  # 加载模板文件

    print_layouts(prs)  # 打印模板中的布局
    # 初始化 LayoutManager，使用配置文件中的 layout_mapping
    layout_manager = LayoutManager(config.layout_mapping)

    # 调用 parse_input_text 函数，解析输入文本，生成 PowerPoint 数据结构
    powerpoint_data, presentation_title = parse_input_text(input_text, layout_manager)

    LOG.info(
        f"解析转换后的 ChatPPT PowerPoint 数据结构:\n{powerpoint_data}"
    )  # 记录调试日志，打印解析后的 PowerPoint 数据

    # 定义输出 PowerPoint 文件的路径
    output_pptx = f"outputs/{presentation_title}.pptx"

    # 调用 generate_presentation 函数生成 PowerPoint 演示文稿
    generate_presentation(powerpoint_data, config.ppt_template, output_pptx)

    return output_pptx


def generate_ppt_button_click():
    """点击按钮时生成 PowerPoint 幻灯片文件"""
    pptx_file = generate_pptx_from_content()
    return pptx_file


# 设置 Gradio 界面
with gr.Blocks() as demo:
    # 创建聊天界面
    chatbot = gr.ChatInterface(fn=generate_markdown_from_input, type="messages")

    # 设置生成 PPT 按钮
    generate_button = gr.Button("生成 PowerPoint 文件")
    pptx_file_output = gr.File(label="下载生成的 PowerPoint 文件")

    # 绑定按钮点击事件
    generate_button.click(fn=generate_ppt_button_click, outputs=pptx_file_output)

demo.launch()
