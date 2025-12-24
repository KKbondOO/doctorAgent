import time
import uuid
import gradio as gr
import os
from minio import Minio, S3Error
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
# Import our graph and db functions
from bot_graph import graph
from db import get_all_threads, delete_thread, get_all_threads_with_titles, setup_title_table

import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = "med-agent"
os.environ["LANGSMITH_ENDPOINT"]= "https://api.smith.langchain.com"

# MinIO Configuration
MinioClient = Minio(
    endpoint="minio1:9000",
    access_key="ROOTUSER",
    secret_key="CHANGEME123",
    secure=False
)

def extract_minio_object_names_from_messages(messages, bucket_name="medimages"):
    """
    从消息列表中提取所有 MinIO 图片的 object_name。
    图片 URL 格式: http://minio1:9000/medimages/eyes_39d2c382.jpg?X-Amz-...
    """
    import re
    object_names = []
    
    for msg in messages:
        if isinstance(msg, HumanMessage):
            if isinstance(msg.content, list):
                for part in msg.content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        # 解析 URL 提取 object_name
                        # 格式: http://minio1:9000/bucket_name/object_name?params
                        pattern = rf"/{bucket_name}/([^?]+)"
                        match = re.search(pattern, url)
                        if match:
                            object_names.append(match.group(1))
    
    return object_names

def delete_minio_images(bucket_name, object_names):
    """
    删除 MinIO 中的多个图片对象。
    """
    deleted_count = 0
    for object_name in object_names:
        try:
            MinioClient.remove_object(bucket_name, object_name)
            print(f"✅ 已删除 MinIO 图片: {bucket_name}/{object_name}")
            deleted_count += 1
        except S3Error as e:
            print(f"⚠️ 删除 MinIO 图片失败 {object_name}: {e}")
    return deleted_count

def delete_minio_images_for_thread(thread_id, graph):
    """
    删除指定对话中所有关联的 MinIO 图片。
    """
    try:
        config = {"configurable": {"thread_id": thread_id}}
        state = graph.get_state(config)
        
        if state and state.values and state.values.get("messages"):
            messages = state.values.get("messages", [])
            object_names = extract_minio_object_names_from_messages(messages)
            
            if object_names:
                print(f"📷 发现 {len(object_names)} 张图片需要删除: {object_names}")
                deleted = delete_minio_images("medimages", object_names)
                print(f"✅ 成功删除 {deleted}/{len(object_names)} 张图片")
                return deleted
            else:
                print("📷 该对话没有关联的图片")
                return 0
    except Exception as e:
        print(f"⚠️ 删除对话图片时发生错误: {e}")
        return 0

def upload_image_2_minio_sync(bucket_name, image_path=None):
    """Upload image to MinIO and return presigned URL."""
    if not image_path:
        raise ValueError("必须提供 image_path")

    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    random_code = str(uuid.uuid4())[:8]
    object_name = f"{name}_{random_code}{ext}"

    if not MinioClient.bucket_exists(bucket_name):
        MinioClient.make_bucket(bucket_name)

    try:
        MinioClient.fput_object(bucket_name, object_name, image_path, content_type="image/jpeg")
        presigned_url = MinioClient.presigned_get_object(bucket_name, object_name)
        return presigned_url
    except S3Error as e:
        raise Exception(f"MinIO 上传失败: {e}")

def format_graph_messages_to_gradio(messages):
    """Convert LangChain messages to Gradio format."""
    gradio_history = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            # Handle text and image content
            if isinstance(msg.content, list):
                # Complex content (text + image)
                for part in msg.content:
                    if part.get("type") == "text":
                        gradio_history.append({"role": "user", "content": part["text"]})
                    elif part.get("type") == "image_url":
                        gradio_history.append({"role": "user", "content": f"![]({part['image_url']['url']})"})
            else:
                gradio_history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            gradio_history.append({"role": "assistant", "content": msg.content})
    return gradio_history

def get_history_options():
    """Get list of thread titles and IDs for radio list."""
    threads_with_titles = get_all_threads_with_titles()
    if not threads_with_titles:
        return gr.Radio(choices=[], value=None)
    
    # Create choices as list of tuples (label, value)
    # Gradio Radio: choices can be list of tuples (label, value)
    choices = [(title, thread_id) for thread_id, title in threads_with_titles]
    default_value = threads_with_titles[0][0] if threads_with_titles else None  # First thread_id
    return gr.Radio(choices=choices, value=default_value)

def load_first_chat():
    """Load the first chat on startup."""
    threads_with_titles = get_all_threads_with_titles()
    if threads_with_titles:
        first_thread_id, first_title = threads_with_titles[0]
        history = load_chat_history(first_thread_id)
        choices = [(title, thread_id) for thread_id, title in threads_with_titles]
        return history, first_thread_id, gr.Radio(choices=choices, value=first_thread_id)
    return [], str(uuid.uuid4()), gr.Radio(choices=[], value=None)

def load_chat_history(thread_id):
    """Load history for a specific thread."""
    if not thread_id:
        return []
    
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)
    if state.values:
        return format_graph_messages_to_gradio(state.values.get("messages", []))
    return []

def new_chat():
    """Start a new chat session."""
    new_id = str(uuid.uuid4())
    return new_id, [], gr.Radio(value=None)

def add_message(history, message):
    global last_user_message
    last_user_message = message

    """Add user message to UI immediately."""
    files = message.get("files") or []
    text = message.get("text") or ""

    if files:
        for x in files:
            history.append({"role": "user", "content": {"path": x}})
    
    if text.strip():
        history.append({"role": "user", "content": text})

    return history, gr.MultimodalTextbox(value=None, interactive=False)

def bot_response(history,thread_id):
    global last_user_message
    message = last_user_message
    """Process message with LangGraph and stream response."""
    if not thread_id:
        thread_id = str(uuid.uuid4())
    
    files = message.get("files") or []
    text = message.get("text") or ""
    
    # Prepare content for LangChain
    content_parts = []
    if text.strip():
        content_parts.append({"type": "text", "text": text})
    else:
        content_parts.append({"type": "text", "text": "解释这张医学图片"})
    
    if files:
        try:
            image_url = upload_image_2_minio_sync("medimages", files[0])
            content_parts.append({"type": "image_url", "image_url": {"url": image_url}})
        except Exception as e:
            error_msg = f"图片上传失败: {e}"
            history.append({"role": "assistant", "content": error_msg})
            yield history, thread_id
            return

    # Create HumanMessage
    human_msg = HumanMessage(content=content_parts)
    
    config = {"configurable": {"thread_id": thread_id}}
    
    # Append empty assistant message for streaming
    history.append({"role": "assistant", "content": ""})
    full_response = ""
    for msg, metadata in graph.stream({"messages": [human_msg]}, config=config, stream_mode="messages"):
        # 只处理来自 chatbot 节点的消息,忽略 save_title 节点中的 LLM 调用
        if metadata.get("langgraph_node") == "chatbot" and isinstance(msg, AIMessage) and msg.content:
            # msg.content 可能是 str，也可能是 list
            if isinstance(msg.content, list):
                parts = []
                for part in msg.content:
                    if isinstance(part, str):
                        parts.append(part)
                content_str = "".join(parts)
            else:
                content_str = msg.content
            full_response += content_str
            history[-1]["content"] = full_response
            yield history, thread_id
            time.sleep(0.01)

def refresh_history_list():
    """Refresh the history list choices without changing the selected value."""
    threads_with_titles = get_all_threads_with_titles()
    choices = [(title, thread_id) for thread_id, title in threads_with_titles]
    return gr.Radio(choices=choices)

def delete_current_chat_thread(thread_id):
    """Delete current chat from database after confirmation."""
    if thread_id:
        # 先删除 MinIO 中关联的图片
        delete_minio_images_for_thread(thread_id, graph)
        # 传递 graph 实例以进行完整的消息删除
        success = delete_thread(thread_id, graph=graph,deleteThread=True)
        if success:
            # Return empty chat, new thread_id, and refreshed history list
            new_id = str(uuid.uuid4())
            threads_with_titles = get_all_threads_with_titles()
            choices = [(title, tid) for tid, title in threads_with_titles]
            return [], new_id, gr.Radio(choices=choices, value=None)
    threads_with_titles = get_all_threads_with_titles()
    choices = [(title, tid) for tid, title in threads_with_titles]
    return [], thread_id, gr.Radio(choices=choices, value=None)

def delete_current_chat_state(thread_id):
    """Delete current chat from database after confirmation."""
    if thread_id:
        # 先删除 MinIO 中关联的图片
        delete_minio_images_for_thread(thread_id, graph)
        # 传递 graph 实例以进行完整的消息删除
        success = delete_thread(thread_id, graph=graph,deleteThread=False)
        if success:
            # Return empty chat, new thread_id, and refreshed history list
            threads_with_titles = get_all_threads_with_titles()
            choices = [(title, tid) for tid, title in threads_with_titles]
            return [], thread_id, gr.Radio(choices=choices, value=None)
    threads_with_titles = get_all_threads_with_titles()
    choices = [(title, tid) for tid, title in threads_with_titles]
    return [], thread_id, gr.Radio(choices=choices, value=None)


# -----------------------------
# Gradio UI
# -----------------------------
css = """
#chatbot { height: 650px !important; }
.gradio-container { font-family: 'Segoe UI', sans-serif; }
.message.user { background: #e8f2ff !important; }
.message.bot { background: #f0f4f9 !important; }
/* 历史列表项样式 - 固定宽度并显示省略号 */
.radio-group label {
    max-width: 250px !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    white-space: nowrap !important;
    display: block !important;
}
"""

with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    
    # State to hold current thread_id
    thread_id_state = gr.State(value=lambda: str(uuid.uuid4()))

    gr.Markdown("""
    <h2 style='text-align: center; color: #2c6efc;'>🔥 MedGemma 医学对话助手 (LangGraph + Postgres)</h2>
    """)

    with gr.Row():
        with gr.Sidebar():
            gr.Markdown("### 📜 历史对话")
            with gr.Column():
                with gr.Row():
                    refresh_btn = gr.Button("🔄 刷新", scale=1)
                    new_chat_btn = gr.Button("➕ 新对话", scale=1)
                history_list = gr.Radio(
                    label="",
                    choices=[],
                    interactive=True,
                    show_label=False
                )
                delete_btn = gr.Button("🗑️ 删除当前对话", variant="stop")
        
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                type="messages",
                height=650,
                label="当前对话",
                show_copy_button=True
            )
            chat_input = gr.MultimodalTextbox(
                interactive=True,
                file_count="single",
                placeholder="请输入问题或上传医学图片…",
                show_label=False,
                sources=["upload"]
            )

    # Events
    
    # 1. Submit Message
    # First add user message to UI
    chat_msg = chat_input.submit(
        add_message,
        inputs=[chatbot, chat_input],
        outputs=[chatbot, chat_input]
    )
    
    # Then call bot response
    bot_msg = chat_msg.then(
        bot_response,
        inputs=[chatbot, thread_id_state],
        outputs=[chatbot, thread_id_state],
        queue=True
    )
    
    # Re-enable input and refresh history list
    bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])
    # Refresh history after bot response to show new titles
    bot_msg.then(refresh_history_list, outputs=[history_list])
    
    # 2. History Management
    # Load first chat on startup
    demo.load(
        load_first_chat,
        outputs=[chatbot, thread_id_state, history_list]
    )
    
    # Refresh list (only updates choices, keeps current selection)
    refresh_btn.click(refresh_history_list, outputs=[history_list])
    
    # Load selected history
    history_list.change(
        load_chat_history,
        inputs=[history_list],
        outputs=[chatbot]
    ).then(
        lambda x: x, inputs=[history_list], outputs=[thread_id_state] # Update thread_id state
    )
    
    # New Chat
    new_chat_btn.click(
        new_chat,
        outputs=[thread_id_state, chatbot, history_list]
    )
    
    # 3. Delete Chat
    delete_btn.click(
        delete_current_chat_thread,
        inputs=[thread_id_state],
        outputs=[chatbot, thread_id_state, history_list]
    )

    chatbot.clear(
        delete_current_chat_state,
        inputs=[thread_id_state],
        outputs=[chatbot, thread_id_state, history_list]
    )

demo.queue()
demo.launch()