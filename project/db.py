import os
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

# Connection string
DB_URI = "postgresql://postgres:changeme123@localhost:5432/med_bot_db"

# Connection Pool
pool = ConnectionPool(conninfo=DB_URI, max_size=20)


# Initialize Store and Checkpointer globally
store = PostgresStore(pool)
checkpointer = PostgresSaver(pool)


def get_all_threads():
    """
    Retrieve a list of all unique thread_ids from the checkpoints.
    This is used to populate the history sidebar.
    """
    # PostgresSaver stores checkpoints in a table (default 'checkpoints')
    # We query for distinct thread_ids.
    # Note: The table name is configurable in PostgresSaver but defaults to 'checkpoints'.
    query = "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id DESC;"

    with pool.connection() as conn:
        with conn.cursor() as cur:
            # Check if table exists first to avoid errors on fresh start
            cur.execute("SELECT to_regclass('public.checkpoints');")
            if cur.fetchone()[0] is None:
                return []
            cur.execute(query)
            rows = cur.fetchall()
            return [row[0] for row in rows]

def delete_thread(thread_id, graph=None, deleteThread=True):
    """
    Delete all checkpoints for a specific thread_id.
    This removes the conversation history from the database.
    
    Args:
        thread_id: The thread ID to delete
        graph: Optional LangGraph instance for proper message deletion
    """
    # 如果提供了 graph,先通过 LangGraph 删除所有消息
    if graph:
        try:
            config = {"configurable": {"thread_id": thread_id}}
            state = graph.get_state(config)
            
            if state and state.values and state.values.get("messages"):
                # 获取所有消息 ID 并创建 RemoveMessage 列表
                messages_to_remove = [
                    RemoveMessage(id=REMOVE_ALL_MESSAGES)
                ]
                
                # 通过 graph 更新状态来删除消息
                graph.update_state(config, {"messages": messages_to_remove})
                print(f"✅ 已通过 LangGraph 删除 thread {thread_id} 的所有消息")
        except Exception as e:
            print(f"⚠️ 通过 LangGraph 删除消息失败: {e}")
            # 继续执行数据库删除
    if deleteThread:
        # 然后删除数据库中的 checkpoints 和 titles
        query_checkpoints = "DELETE FROM checkpoints WHERE thread_id = %s;"
        query_titles = "DELETE FROM conversation_titles WHERE thread_id = %s;"
        
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query_checkpoints, (thread_id,))
                cur.execute(query_titles, (thread_id,))
                conn.commit()
                print(f"✅ 已从数据库删除 thread {thread_id} 的记录")
                return cur.rowcount > 0
    namespace = ("user_prefs", thread_id)
    key = "style"
    store.delete(namespace, key)

def setup_title_table():
    """
    Create the conversation_titles table if it doesn't exist.
    """
    create_table_query = """
    CREATE TABLE IF NOT EXISTS conversation_titles (
        thread_id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(create_table_query)
            conn.commit()

def save_conversation_title(thread_id, title):
    """
    Save or update the title for a conversation thread.
    """
    query = """
    INSERT INTO conversation_titles (thread_id, title, updated_at)
    VALUES (%s, %s, CURRENT_TIMESTAMP)
    ON CONFLICT (thread_id)
    DO UPDATE SET title = EXCLUDED.title, updated_at = CURRENT_TIMESTAMP;
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (thread_id, title))
            conn.commit()

def get_conversation_title(thread_id):
    """
    Get the title for a specific conversation thread.
    Returns the title or None if not found.
    """
    query = "SELECT title FROM conversation_titles WHERE thread_id = %s;"
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (thread_id,))
            result = cur.fetchone()
            return result[0] if result else None

def get_all_threads_with_titles():
    """
    Retrieve all threads with their titles.
    Returns a list of (thread_id, title) tuples, ordered by most recent first.
    """
    query = """
    SELECT DISTINCT c.thread_id, 
           COALESCE(t.title, c.thread_id) as display_title,
           MAX(t.created_at) as last_updated
    FROM checkpoints c
    LEFT JOIN conversation_titles t ON c.thread_id = t.thread_id
    GROUP BY c.thread_id, t.title
    ORDER BY last_updated DESC NULLS LAST, c.thread_id DESC;
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            # Check if table exists first
            cur.execute("SELECT to_regclass('public.checkpoints');")
            if cur.fetchone()[0] is None:
                return []
            cur.execute(query)
            rows = cur.fetchall()
            return [(row[0], row[1]) for row in rows]

if __name__ == "__main__":
    print("Testing database connection...")
    try:
        # Ensure tables exist
        setup_title_table()
        print("Tables setup successfully.")
        
        threads = get_all_threads()
        print(f"Connection successful. Found {len(threads)} threads.")
    except Exception as e:
        print(f"Error connecting to database: {e}")
