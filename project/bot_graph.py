from typing import Annotated, Literal, TypedDict, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, Secret, SecretStr
from db import checkpointer, store, save_conversation_title, get_conversation_title
import re
import ollama
import numpy as np
from langchain_core.messages.utils import (  
    trim_messages,  
    count_tokens_approximately  
)
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = "doctor-agent"
os.environ["LANGSMITH_ENDPOINT"]= "https://api.smith.langchain.com"

# 配置参数
MILVUS_HOST = "localhost"  # Milvus服务地址
MILVUS_PORT = "19530"      # Milvus服务端口
COLLECTION_NAME = "Medical_QA_Pairs"  # 集合名称
VECTOR_DIM = 1024            # 向量维度（与Schema一致）
MODEL_NAME = "qwen3-embedding:0.6b"

# Define State
class State(TypedDict):
    messages: Annotated[list, add_messages]
    decision: str
    department: str
    keywords: list[str]
    ragMessage: str
    imageAnalysis: str  # 存储图片分析结果

class Route(BaseModel):
    step:Literal["rag","normal"] = Field(None, description="选择执行RAG还是正常对话")  # pyright: ignore[reportAssignmentType]
class MedicalQueryInfo(BaseModel):
    department: Optional[str] = Field(None, description="科室信息")
    keywords: Optional[list[str]] = Field(None, description="关键词合集")

# Initialize LLM (vLLM compatible)
llm = ChatOpenAI(
    api_key=SecretStr("unsloth/medgemma-4b-it-bnb-4bit"),
    base_url="http://localhost:8000/v1",
    model="unsloth/medgemma-4b-it-bnb-4bit",
    temperature=0.9,
    max_tokens=2048,
)
llm_qwen = ChatOpenAI(
    api_key=SecretStr("Qwen/Qwen3-0.6B"),
    base_url="http://localhost:8900/v1",
    model="Qwen/Qwen3-0.6B",
    temperature=0.8,
    max_tokens=512,
)
router = llm_qwen.with_structured_output(Route)
generator = llm_qwen.with_structured_output(MedicalQueryInfo)

# 初始化Milvus客户端（假设已经配置好连接）
from pymilvus import MilvusClient
# 创建Milvus客户端实例
client = MilvusClient(
    uri="http://"+MILVUS_HOST+":"+MILVUS_PORT # 根据实际情况修改
)

# 使用 qwen3:0.6b 生成标题
def generate_title_with_ollama(user_question, ai_response):
    """
    使用 Ollama 的 qwen3:0.6b 模型生成对话标题
    """
    prompt = f"根据对话生成标题(10字以内):用户问:{user_question[:64]} AI答:{ai_response[:512]}"
    try:
        response = llm_qwen.invoke(prompt)
        content = response.content
        title = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        if len(title) > 15:
            title = title[:15] + "..."
        return title if title else "新对话"
    except Exception as e:
        return "新对话"

def llm_call_router(state: State):
    """Route the input to the appropriate node"""
    # 查找最新的 HumanMessage
    messages = state["messages"]
    latest_human_message = None
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            latest_human_message = message
            break
    if latest_human_message is None:
        # 处理没有找到 HumanMessage 的情况
        raise ValueError("No HumanMessage found in messages")
    
    # 检测消息中是否包含图片（多模态消息）
    has_image = False
    if isinstance(latest_human_message.content, list):
        for part in latest_human_message.content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                has_image = True
                break
    
    # 如果包含图片，默认选择RAG以提供医学建议
    if has_image:
        print("检测到医学图片，默认执行RAG检索以提供医学建议")
        return {"decision": "rag"}
    
    # 否则通过路由器判断
    decision = router.invoke([
        SystemMessage(content="根据用户的医疗相关问题，判断是否需要检索专业知识库(RAG)来回答。如果是医疗专业问题，选择'rag'；如果是日常对话或非专业医疗问题，选择'normal'。"),
        latest_human_message
    ])
    print("路由结果：",decision)
    return {"decision": decision.step}

def llm_generate_department_keyword(state: State):
    """根据用户问题或图片分析结果识别科室和关键词"""
    # 查找最新的 HumanMessage
    messages = state["messages"]
    latest_human_message = None
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            latest_human_message = message
            break
    if latest_human_message is None:
        raise ValueError("No HumanMessage found in messages")
    
    # 优先使用图片分析结果（如果存在）
    image_analysis = state.get("imageAnalysis", "")
    if image_analysis and image_analysis.strip():
        # 基于图片分析结果提取关键词
        user_text = image_analysis
        print("使用图片分析结果提取科室和关键词")
    else:
        # 提取文本内容（支持多模态消息）
        user_text = ""
        if isinstance(latest_human_message.content, str):
            user_text = latest_human_message.content
        elif isinstance(latest_human_message.content, list):
            # 从多模态消息中提取文本部分
            for part in latest_human_message.content:
                if isinstance(part, dict) and part.get("type") == "text":
                    user_text += part.get("text", "")
        
        # 如果没有文本内容（纯图片消息），使用默认关键词
        if not user_text.strip():
            user_text = "医学影像分析诊断建议"
    
    # 定义可用科室列表
    departments = [
        "乳腺科", "产科", "内分泌科", "呼吸科", "心外科", "心血管科",
        "感染科", "普通内科", "普通外科", "泌尿科", "消化科", "神经科",
        "神经脑外科", "肛肠", "肝病科", "肝胆科", "肾内科", "胸外科",
        "血液科", "血管科", "风湿免疫科","眼科"
    ]
    # 构造提示词
    prompt = f"""
    根据用户的医疗问题或图片分析结果，完成以下任务：
    1. 从以下科室列表中选择最匹配的一个科室：
    {', '.join(departments)}
    2. 提取问题中的最多5个关键医学术语作为关键词
    用户问题/图片分析：{user_text}
    """
    # 调用模型生成科室和关键词
    result = generator.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=user_text)
    ])
    print("科室和关键词：",result)
    return {
        "department": result.department,
        "keywords": result.keywords
    }

def image_analyzer(state: State, config):
    """
    分析图片内容，生成初步的医学影像解释
    这个解释将用于后续的RAG检索
    """
    messages = state["messages"]
    
    # 查找最新的包含图片的 HumanMessage
    latest_human_message = None
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            latest_human_message = message
            break
    
    if latest_human_message is None:
        raise ValueError("No HumanMessage found in messages")
    
    # 构造图片分析提示词
    system_prompt = """你是一位专业的医学影像分析助手。请仔细分析用户提供的医学图片，并提供详细的影像描述，包括：1. 图片类型（X光、CT、MRI等）2. 观察到的主要特征3. 可能的异常或值得注意的地方4. 初步的医学判断请用专业但易懂的语言描述，不需要给出最终诊断建议（建议将在后续步骤生成）。"""
    
    # 准备消息
    messages_for_analysis = [
        SystemMessage(content=system_prompt),
        latest_human_message
    ]
    
    # 调用模型分析图片
    print("正在分析医学图片...")
    response = llm.invoke(messages_for_analysis)
    analysis_text = response.content
    
    print(f"图片分析结果: {analysis_text[:1024]}...")
    
    # 将分析结果存储到state
    return {
        "imageAnalysis": analysis_text
    }

def rag(state: State):
    """RAG检索函数，逐级过滤信息后进行向量检索"""
    # 从state中提取必要信息
    messages = state["messages"]
    department = state.get("department")
    keywords = " ".join(state.get("keywords", []))
    imageAnalysis = state.get("imageAnalysis")

    # 获取最新的用户消息用于向量检索
    latest_human_message = None
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            latest_human_message = message
            break

    if latest_human_message is None:
        raise ValueError("No HumanMessage found in messages")

    try:
        filter_expr = []
        # 1. 基于department进行分区过滤（使用实际字段名，Milvus会自动进行分区路由优化）
        if department:
            filter_expr.append(f'department == "{department}"')

        # 2. 构建查询条件
        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": 50}
        }

        # 3. 关键词文本匹配过滤
        filter_expr.append(f'TEXT_MATCH(title, "{keywords}")')

        # 4. 向量检索 - 提取消息文本内容
        message_text = ""
        if isinstance(latest_human_message.content, str):
            message_text = latest_human_message.content
            print("用于向量检索的用户消息：",message_text)
        elif isinstance(latest_human_message.content, list):
            # 从多模态消息中提取文本部分
            for part in latest_human_message.content:
                if isinstance(part, dict) and part.get("type") == "text":
                    message_text += part.get("text", "")
            print("用于向量检索的用户消息：",message_text)
        
        prompt = f"Instruct: question answering\nQuery: {message_text}"
        # 生成嵌入
        res = ollama.embeddings(
            model=MODEL_NAME,
            prompt=prompt
        )
        query_vector = res['embedding']
        
        vec_array = np.array(query_vector, dtype=np.float32)

        if imageAnalysis:
            image_prompt = f"Instruct: question answering\nQuery: {message_text} {imageAnalysis}"
            # 生成嵌入
            image_res = ollama.embeddings(
                model=MODEL_NAME,
                prompt=image_prompt
            )
            image_qa_vector = image_res['embedding']
            image_qa_vec_array = np.array(image_qa_vector, dtype=np.float32)
        else:
            image_qa_vec_array = None

        if image_qa_vec_array is None:
            data = [vec_array.tolist()]
        else:
            data = [image_qa_vec_array.tolist()]

        expr = " && ".join(filter_expr)
        print("搜索表达式：",expr)

        # 执行搜索
        # 注意：data参数需要是列表的列表格式，即使只查询一个向量
        search_result = client.search(
            collection_name=COLLECTION_NAME,
            data=data, # 包装成列表的列表
            filter=expr,
            limit=1,
            output_fields=["department","title","answer"],
            search_params=search_params
        )
        print("搜索结果：",search_result)
        # 5. 处理检索结果
        if search_result and len(search_result) > 0:
            # 获取最相关的答案
            top_result = search_result[0][0]  # 第一个查询的第一个结果
            answer = top_result.get("entity", {}).get("answer", "")

            return {
                "ragMessage": answer
            }
        else:
            # 如果没有找到匹配结果，返回默认信息
            return {
                "ragMessage": "抱歉，未找到相关的医学知识。"
            }

    except Exception as e:
        print(f"RAG检索过程中发生错误: {e}")
        return {
            "ragMessage": "检索过程中出现错误，请稍后重试。"
        }

def chatbot(state: State, config):
    """
    Node that calls the LLM.
    """
    # Demonstrate Store usage: Retrieve user preferences
    # We use thread_id as a proxy for user_id in this demo
    thread_id = config["configurable"]["thread_id"]
    namespace = ("user_prefs", thread_id)
    key = "style"
    
    messages = trim_messages(  
    state["messages"],
    strategy="last",
    token_counter=count_tokens_approximately,
    max_tokens=2816,
    start_on="human",
    end_on=("human", "tool"),
    )
    
    # Simple logic to update store (in a real app, use a tool)
    last_msg = messages[-1]
    print(f"消息内容: {last_msg.content if isinstance(last_msg, HumanMessage) else 'N/A'}")
    
    # 检查消息内容中是否包含"简洁"
    contains_concise = False
    if isinstance(last_msg, HumanMessage):
        if isinstance(last_msg.content, str):
            contains_concise = "简洁" in last_msg.content
        elif isinstance(last_msg.content, list):
            # 检查列表中的文本内容
            for part in last_msg.content:
                if isinstance(part, dict) and part.get("type") == "text":
                    if "简洁" in part.get("text", ""):
                        contains_concise = True
                        break
    
    if contains_concise:
        print("检测到'简洁'关键字,使用到了Store")
        store.put(namespace, key, {"style": "concise"})
        
    # Read preference
    stored_pref = store.get(namespace, key)
    system_prompt = "You are a helpful medical assistant."
    if stored_pref and stored_pref.value.get("style") == "concise":
        system_prompt += " Please be very concise."
    # 检查是否包含图片（多模态消息）
    has_image = False
    if isinstance(last_msg, HumanMessage) and isinstance(last_msg.content, list):
        for part in last_msg.content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                has_image = True
                break
    
    # 检查是否有图片分析结果和RAG检索结果
    image_analysis = state.get("imageAnalysis", "")
    rag_message = state.get("ragMessage", "")
    
    # 根据消息类型处理RAG结果
    if has_image and image_analysis and image_analysis.strip():
        # 图片消息的第二阶段：基于图片分析和RAG结果生成综合建议
        print(f"图片分析+RAG模式：生成综合医学建议")
        system_prompt = """你是一位专业的医学助手。现在你需要基于以下信息为用户提供综合的医学建议：

        1. 你之前对医学图片的分析结果
        2. 从医学知识库检索到的相关医学问答对

        请整合这些信息，为用户提供：
        - 对图片的专业解读
        - 基于知识库的相关医学知识
        - 具体的医学建议和注意事项
        - 是否需要进一步检查或就医的建议

        请用专业但易懂的语言，给出全面的医学建议。"""
        
        # 构造包含分析结果和RAG信息的消息
        combined_info = f"""**医学图片分析结果：**{image_analysis}"""
        if rag_message and rag_message.strip():
            combined_info += f"""**医学知识库参考信息：**{rag_message}"""
        combined_info += "请基于以上信息，为用户提供综合的医学建议。"
        
        # 使用文本消息传递综合信息
        final_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=combined_info)
        ]
        
        print("最终传给模型的消息（图片综合建议）", final_messages)
        response = llm.invoke(final_messages)
        return {"messages": [response]}
        
    elif not has_image and rag_message and rag_message.strip():
        # 普通文本消息：将RAG结果融入系统提示词
        print(f"检测到文本消息，进行检索增强生成")
        system_prompt += f"""以下是从医学知识库中检索到的相关医学问答对，请参考这些信息回答用户的问题：参考答案：{rag_message}请基于以上参考信息，结合你的医学知识，为用户提供准确、专业的回答。如果参考信息与用户问题不完全匹配，请根据医学常识进行适当的扩展和说明。"""
        
    # Sanitize messages: Merge consecutive HumanMessages
    sanitized_messages = []
    for msg in messages:
        if sanitized_messages and isinstance(msg, HumanMessage) and isinstance(sanitized_messages[-1], HumanMessage):
            # Merge content
            last_msg = sanitized_messages[-1]
            if isinstance(last_msg.content, str) and isinstance(msg.content, str):
                last_msg.content += "\n" + msg.content
            elif isinstance(last_msg.content, list) and isinstance(msg.content, str):
                last_msg.content.append({"type": "text", "text": msg.content})
            elif isinstance(last_msg.content, str) and isinstance(msg.content, list):
                new_content = [{"type": "text", "text": last_msg.content}]
                new_content.extend(msg.content)
                last_msg.content = new_content
            elif isinstance(last_msg.content, list) and isinstance(msg.content, list):
                last_msg.content.extend(msg.content)
        else:
            sanitized_messages.append(msg)

    # Prepend System Message
    # We don't want to keep adding it to history, so we just pass it to invoke
    final_messages = [SystemMessage(content=system_prompt)] + sanitized_messages
    print("最终传给模型的消息", final_messages)
    response = llm.invoke(final_messages)
    
    return {"messages": [response]}

def check_needTitle(state: State):
    # 检查是否是新对话(只有1轮对话,即2条消息)
    # 统计非系统消息的数量
    messages = state["messages"]
    print("执行check_needTitle")
    human_messages = [m for m in messages if isinstance(m, HumanMessage)]
    if len(human_messages) == 1:  # 这是第一个用户消息
        print("确认生成标题")
        return "True"
    return "False"

def save_title(state: State,config):
    # 在生成响应后,生成并保存标题
    # 注意: 这个节点不返回任何状态更新,避免触发流式输出
    try:
        # 提取用户问题
        user_question = ""
        ai_response = ""
        messages = state["messages"]
        thread_id = config["configurable"]["thread_id"]
        
        for msg in messages:
            if isinstance(msg, HumanMessage):
                if isinstance(msg.content, str):
                    user_question = msg.content
                elif isinstance(msg.content, list):
                    # 提取文本内容
                    for part in msg.content:
                        if part.get("type") == "text":
                            user_question += part["text"]
            elif isinstance(msg, AIMessage):
                if isinstance(msg.content, str):
                    ai_response = msg.content
                elif isinstance(msg.content, list):
                    # 提取文本内容
                    for part in msg.content:
                        if part.get("type") == "text":
                            ai_response += part["text"]
        print("用户问题:", user_question)
        print("AI回复:", ai_response)
        # 生成标题
        if user_question and ai_response:
            title = generate_title_with_ollama(user_question, ai_response)
            # 保存标题到数据库
            save_conversation_title(thread_id, title)
            print(f"为对话 {thread_id} 生成标题: {title}")
    except Exception as e:
        print(f"保存标题失败: {e}")
    
    # 明确返回空字典,不更新状态,避免触发流式输出
    return {}

def route_decision(state: State):
    # Return the node name you want to visit next
    decision = state["decision"]
    
    # 检查是否是图片消息
    messages = state["messages"]
    latest_human_message = None
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            latest_human_message = message
            break
    
    has_image = False
    if latest_human_message and isinstance(latest_human_message.content, list):
        for part in latest_human_message.content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                has_image = True
                break
    
    # 如果是图片消息且需要RAG，先进行图片分析
    if has_image and decision == "rag":
        return "image_analyzer"
    # 如果是文本消息且需要RAG，直接进行科室关键词提取
    elif decision == "rag":
        return "llm_generate_department_keyword"
    # 普通对话直接进入chatbot
    else:
        return "chatbot"

# Build Graph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("save_title", save_title)
graph_builder.add_node("llm_call_router",llm_call_router)
graph_builder.add_node("image_analyzer", image_analyzer)  # 新增图片分析节点
graph_builder.add_node("llm_generate_department_keyword",llm_generate_department_keyword)
graph_builder.add_node("rag", rag)


# 修改图的执行流程：
# 文本消息: START → llm_call_router → (rag流程 或 直接对话) → chatbot → 保存标题
# 图片消息: START → llm_call_router → image_analyzer → llm_generate_department_keyword → rag → chatbot → 保存标题
graph_builder.add_edge(START, "llm_call_router")
graph_builder.add_conditional_edges(
    "llm_call_router", 
    route_decision, 
    {
        "image_analyzer": "image_analyzer",  # 图片消息先分析图片
        "llm_generate_department_keyword": "llm_generate_department_keyword",  # 文本消息RAG流程
        "chatbot": "chatbot"  # 普通对话
    }
)
graph_builder.add_edge("image_analyzer", "llm_generate_department_keyword")  # 图片分析后提取关键词
graph_builder.add_edge("llm_generate_department_keyword", "rag")
graph_builder.add_edge("rag", "chatbot")
graph_builder.add_conditional_edges("chatbot", check_needTitle, {"True": "save_title", "False": END})
graph_builder.add_edge("save_title", END)

# Compile Graph with Checkpointer
graph = graph_builder.compile(checkpointer=checkpointer)

