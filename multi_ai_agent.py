import streamlit as st
import json
import os
import operator
import logging
import uuid
import asyncio
from dotenv import load_dotenv, find_dotenv
from typing import List, Annotated, TypedDict

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage, messages_to_dict, messages_from_dict
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='agent_conversation.log',
    filemode='a'
)
logger = logging.getLogger(__name__)

MODEL_PRICING_PER_MILLION_TOKENS = {
    "gemini-2.0-flash": {
        "input": 0.35,
        "output": 0.70
    },
    "default": {
        "input": 0.35,
        "output": 0.70
    }
}

def calculate_cost(usage_metadata: dict, model_name: str) -> dict:
    if not usage_metadata:
        return {"input": 0.0, "output": 0.0, "total": 0.0}
    pricing = MODEL_PRICING_PER_MILLION_TOKENS.get(model_name, MODEL_PRICING_PER_MILLION_TOKENS["default"])
    input_tokens = usage_metadata.get("prompt_token_count", 0)
    output_tokens = usage_metadata.get("candidates_token_count", 0)
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    total_cost = input_cost + output_cost
    return {"input": input_cost, "output": output_cost, "total": total_cost}

def run_async_in_sync(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

def sanitize_schema(item):
    if isinstance(item, dict):
        item.pop('additionalProperties', None)
        item.pop('$schema', None)
        if 'type' in item and isinstance(item['type'], list):
            non_null_types = [t for t in item['type'] if str(t).upper() != 'NULL']
            item['type'] = str(non_null_types[0]).upper() if non_null_types else None
        for key, value in item.items():
            item[key] = sanitize_schema(value)
    elif isinstance(item, list):
        return [sanitize_schema(i) for i in item]
    return item

_ = load_dotenv(find_dotenv())
google_api_key = os.getenv("GOOGLE_API_KEY")

CONVERSATION_HISTORY_DIR = "conversation_history"
os.makedirs(CONVERSATION_HISTORY_DIR, exist_ok=True)

def save_conversation(session_id: str, messages: List[BaseMessage]):
    """会話をJSONファイルに保存する"""
    if not session_id or not messages:
        return
    file_path = os.path.join(CONVERSATION_HISTORY_DIR, f"{session_id}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(messages_to_dict(messages), f, ensure_ascii=False, indent=2)

def load_conversation(session_id: str) -> List[BaseMessage]:
    """JSONファイルから会話を読み込む"""
    file_path = os.path.join(CONVERSATION_HISTORY_DIR, f"{session_id}.json")
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            return messages_from_dict(data)
        except (json.JSONDecodeError, TypeError):
            return []

def list_conversations() -> List[dict]:
    """保存されている会話の一覧を取得する"""
    conversations = []
    for filename in os.listdir(CONVERSATION_HISTORY_DIR):
        if filename.endswith(".json"):
            session_id = filename[:-5]
            file_path = os.path.join(CONVERSATION_HISTORY_DIR, filename)
            try:
                mtime = os.path.getmtime(file_path)
                messages = load_conversation(session_id)
                first_user_message = next((m.content for m in messages if isinstance(m, HumanMessage) and m.additional_kwargs.get("role") != "internal_instruction"), "新しい会話")
                title = first_user_message[:40] + "..." if len(first_user_message) > 40 else first_user_message
                conversations.append({"id": session_id, "title": title, "mtime": mtime})
            except Exception:
                continue
    conversations.sort(key=lambda x: x["mtime"], reverse=True)
    return conversations

def delete_conversation(session_id: str):
    """会話ファイルを削除する"""
    file_path = os.path.join(CONVERSATION_HISTORY_DIR, f"{session_id}.json")
    if os.path.exists(file_path):
        os.remove(file_path)

class AgentState(TypedDict):
    messages: List[BaseMessage]
    next: str

def create_worker(llm: ChatGoogleGenerativeAI, tools: list, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])
    return prompt | llm.bind_tools(tools)

def create_supervisor(llm: ChatGoogleGenerativeAI, worker_names: List[str]):
    system_prompt = (
        "あなたはAIチームのマネージャーです。あなたの仕事は、ユーザーの要求を達成するために、部下であるワーカーチームを監督することです。\n"
        "会話の履歴全体（ユーザーの要求、ワーカーのこれまでの作業結果など）を注意深く確認してください。\n\n"
        "以下の手順で行動してください:\n"
        "1. **タスクの分析**: ユーザーの要求を達成するために必要なステップを考えます。複数のワーカーによる連携が必要な場合もあります。例えば、'Webサーファー'が収集した情報を'ファイルオペレーター'がファイルに書き込む、といった連携です。\n"
        "2. **次の行動の決定**: 分析に基づき、次に取るべき行動を決定します。\n"
        "   - **ワーカーへの指示**: 特定のタスクをワーカーに任せる場合、そのワーカーの名前を`next`に指定し、具体的な指示内容を`content`に記述します。**重要なのは、以前のワーカーの出力結果を、次のワーカーへの指示に含めることです。** これにより、ワーカー間で情報を引き継ぐことができます。\n"
        "   - **ユーザーへの直接回答**: 全てのタスクが完了した場合、またはワーカーを必要としない単純な応答の場合は、`next`に'FINISH'を指定し、ユーザーへの最終的な回答を`content`に記述します。\n"
        "   - **失敗からの回復**: ワーカーがタスクに失敗した場合は、会話履歴を確認し、指示内容を修正して再試行するか、別のアプローチを検討してください。\n\n"
        f"利用可能なワーカー:\n{chr(10).join(f'- {name}' for name in worker_names)}"
    )
    output_schema = {
        "title": "supervisor_decision",
        "type": "object",
        "properties": {
            "next": {"type": "string", "description": f"次に実行するワーカー名（{', '.join(worker_names)} または FINISH）"},
            "content": {"type": "string", "description": "ワーカーへの指示内容、またはユーザーへの最終回答"}
        },
        "required": ["next", "content"]
    }
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])
    llm_with_tool = llm.bind_tools(tools=[output_schema], tool_choice="supervisor_decision")
    return prompt, llm_with_tool

@st.cache_resource
def initialize_graph():
    supervisor_model_name = "gemini-2.0-flash"
    supervisor_llm_instance = ChatGoogleGenerativeAI(model=supervisor_model_name, temperature=0.0, google_api_key=google_api_key)
    worker_model_name = "gemini-2.0-flash"
    worker_llm_instance = ChatGoogleGenerativeAI(model=worker_model_name, temperature=0.0, google_api_key=google_api_key)
    
    with open("mcp_config.json", "r") as f:
        mcp_config = json.load(f)
    mcp_client = MultiServerMCPClient(mcp_config["mcpServers"])
    
    tools = run_async_in_sync(mcp_client.get_tools())
    sanitized_tools = [sanitize_schema(convert_to_openai_function(t)) for t in tools]
    
    workers = {
        "Webサーファー": create_worker(worker_llm_instance, sanitized_tools, "あなたはWeb検索の専門家です。web-searchツールを使用することができます。\n与えられた指示を達成するために適切な検索ワードを考え、必要な情報を検索してください。検索結果は、次の担当者（または最終的な回答者）が理解しやすいように、明確かつ詳細に報告してください。"),
        "ファイルオペレーター": create_worker(worker_llm_instance, sanitized_tools, "あなたはローカルファイルを操作する専門家です。file-systemツールを使用することができます。\n与えられた指示（ファイルパスや書き込む内容など）に正確に従って、ファイル操作を実行してください。操作が成功したか、失敗したかを明確に報告してください。"),
    }
    
    supervisor_prompt, supervisor_llm = create_supervisor(supervisor_llm_instance, list(workers.keys()))

    def supervisor_node(state: AgentState):
        logger.info("--- Supervisor Node ---")
        logger.info(f"Input State: {state['messages']}")
        chain = supervisor_prompt | supervisor_llm
        response_message = chain.invoke({"messages": state["messages"]})
        usage_metadata = response_message.response_metadata.get("usage_metadata", {})
        costs = calculate_cost(usage_metadata, supervisor_model_name)
        logger.info(f"Cost (Supervisor - {supervisor_model_name}): Input: ${costs['input']:.6f}, Output: ${costs['output']:.6f}, Total: ${costs['total']:.6f}")
        logger.info(f"Token Usage (Supervisor): {usage_metadata}")
        tool_call = response_message.tool_calls[0]
        supervisor_output = tool_call['args']
        logger.info(f"Output: {supervisor_output}")
        content = supervisor_output.get("content", "")
        next_action = supervisor_output.get("next", "FINISH")
        supervisor_comment_content = content if next_action == "FINISH" else f"【指示: {next_action}へ】\n{content}"
        supervisor_comment = AIMessage(content=supervisor_comment_content, name="Supervisor")
        if next_action != "FINISH":
            instruction_for_worker = HumanMessage(content=content, additional_kwargs={"role": "internal_instruction"})
            return {"messages": state["messages"] + [supervisor_comment, instruction_for_worker], "next": next_action}
        else:
            return {"messages": state["messages"] + [supervisor_comment], "next": next_action}

    def worker_node(state: AgentState):
        worker_name = state["next"]
        logger.info(f"--- Worker Node: {worker_name} ---")
        messages_for_worker = state['messages']
        logger.info(f"Input Messages for Worker: {messages_for_worker}")
        worker = workers[worker_name]
        response = worker.invoke({"messages": messages_for_worker}, {"recursion_limit": 10})
        usage_metadata = response.response_metadata.get("usage_metadata", {})
        costs = calculate_cost(usage_metadata, worker_model_name)
        logger.info(f"Cost ({worker_name} - {worker_model_name}): Input: ${costs['input']:.6f}, Output: ${costs['output']:.6f}, Total: ${costs['total']:.6f}")
        logger.info(f"Token Usage ({worker_name}): {usage_metadata}")
        logger.info(f"Output: {response}")
        finish_reason = response.response_metadata.get('finish_reason', '')
        if finish_reason == 'MALFORMED_FUNCTION_CALL' or (not response.content and not hasattr(response, 'tool_calls')):
            error_message = f"（システムエラー：{worker_name}がタスクの実行に失敗しました。理由: {finish_reason}。指示を修正して再試行します。）"
            logger.error(f"Worker {worker_name} failed. Reason: {finish_reason}. Message: {response.response_metadata.get('finish_message', 'N/A')}")
            response = AIMessage(content=error_message, name=worker_name)
        response.name = worker_name
        return {"messages": state["messages"] + [response]}

    _tool_node = ToolNode(tools)

    async def custom_tool_node(state: AgentState):
        tool_results = await _tool_node.ainvoke(state)
        return {"messages": state["messages"] + tool_results["messages"]}

    def after_worker_router(state: AgentState):
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "supervisor"

    def supervisor_router(state: AgentState):
        next_val = state.get("next")
        if not next_val or next_val == "FINISH":
            return END
        return next_val

    workflow = StateGraph(AgentState)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("tools", custom_tool_node)
    for name in workers:
        workflow.add_node(name, worker_node)
        workflow.add_conditional_edges(name, after_worker_router, {"tools": "tools", "supervisor": "supervisor"})
    workflow.add_edge("tools", "supervisor")
    workflow.add_conditional_edges("supervisor", supervisor_router, {**{name: name for name in workers}, END: END})
    workflow.add_edge(START, "supervisor")
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app

st.set_page_config(page_title="Multi-Agent AI", page_icon="✨", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #f0f2f6; }
    [data-testid="stChatMessage"] { background-color: white; border-radius: 0.75rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 1rem; }
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e6e6e6; }
    .stButton>button { border-radius: 0.5rem; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("🤖 コントロールパネル")
    st.markdown("---")
    if st.button("➕ 新しい会話を開始", use_container_width=True, type="primary"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.header("📜 会話履歴")
    past_conversations = list_conversations()
    if not past_conversations:
        st.caption("まだ会話履歴はありません。")

    for conv in past_conversations:
        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button(conv["title"], key=f"load_{conv['id']}", use_container_width=True):
                st.session_state.session_id = conv["id"]
                st.session_state.messages = load_conversation(conv["id"])
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"delete_{conv['id']}", use_container_width=True, help="この会話を削除します"):
                delete_conversation(conv["id"])
                if st.session_state.get("session_id") == conv["id"]:
                    st.session_state.session_id = str(uuid.uuid4())
                    st.session_state.messages = []
                st.rerun()

st.title("✨ Multi-Agent AI")
st.caption("AIエージェントチームが協調して、あなたのリクエストを処理します。")

if not google_api_key:
    st.error("Google AI StudioのAPIキーが設定されていません。.envファイルに GOOGLE_API_KEY を設定してください。")
    st.stop()

try:
    graph = initialize_graph()
except Exception as e:
    st.error(f"アプリケーションの初期化中にエラーが発生しました: {e}")
    st.exception(e)
    st.stop()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []

def render_internal_message(msg: BaseMessage):
    avatar_map = {"Supervisor": "🤖", "Webサーファー": "🌐", "ファイルオペレーター": "📁", "internal_instruction": "📝", "tool_call": "🛠️", "tool_result": "✅"}
    name, avatar = "System", "⚙️"
    if hasattr(msg, 'name') and msg.name:
        name, avatar = msg.name, avatar_map.get(msg.name, "🕵️")
    elif isinstance(msg, HumanMessage) and msg.additional_kwargs.get("role") == "internal_instruction":
        name, avatar = "内部指示", avatar_map["internal_instruction"]
    elif isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
        name, avatar = "ツール呼び出し", avatar_map["tool_call"]
    elif isinstance(msg, ToolMessage):
        name, avatar = f"ツール実行結果 ({msg.name})", avatar_map["tool_result"]
    with st.chat_message(name, avatar=avatar):
        if isinstance(msg, AIMessage) and not msg.content and hasattr(msg, 'tool_calls') and msg.tool_calls:
            tool_call = msg.tool_calls[0]
            st.info(f"ツール `{tool_call['name']}` を呼び出します。")
            st.code(json.dumps(tool_call['args'], indent=2, ensure_ascii=False), language="json")
        elif isinstance(msg, ToolMessage):
            try:
                content_dict = json.loads(msg.content)
                st.code(json.dumps(content_dict, indent=2, ensure_ascii=False), language="json")
            except (json.JSONDecodeError, TypeError):
                st.code(str(msg.content), language="text")
        else:
            st.markdown(msg.content)

turns = []
current_turn_messages = []
for msg in st.session_state.get("messages", []):
    is_real_user_message = isinstance(msg, HumanMessage) and msg.additional_kwargs.get("role") != "internal_instruction"
    if is_real_user_message and current_turn_messages:
        turns.append(current_turn_messages)
        current_turn_messages = []
    current_turn_messages.append(msg)
if current_turn_messages:
    turns.append(current_turn_messages)

for turn in turns:
    user_message, agent_steps = turn[0], turn[1:]
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_message.content)
    if agent_steps:
        final_answer = None
        internal_steps = []
        last_step = agent_steps[-1]
        if isinstance(last_step, AIMessage) and last_step.name == "Supervisor":
            final_answer = last_step
            internal_steps = agent_steps[:-1]
        else:
            internal_steps = agent_steps
        if internal_steps:
            with st.expander("🧠 エージェントの思考プロセスを見る"):
                for step in internal_steps:
                    render_internal_message(step)
        if final_answer:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(final_answer.content)

if prompt := st.chat_input("Web検索やファイル操作など、何でも聞いてください..."):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    with st.spinner("🧠 AIエージェントが思考中..."):
        try:
            config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            input_messages = {"messages": st.session_state.messages}
            final_state = run_async_in_sync(graph.ainvoke(input_messages, config))
            st.session_state.messages = final_state["messages"]
            save_conversation(st.session_state.session_id, st.session_state.messages)
            st.rerun()
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
            st.exception(e)
