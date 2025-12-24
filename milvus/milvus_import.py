import pandas as pd
import ollama
import numpy as np
import sys
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility
)

# Configuration
# INPUT_FILE = r"C:\Users\27160\Desktop\Chinese-medical-dialogue-data-master\样例_内科5000-6000.csv"
# INPUT_FILE = r"C:\Users\27160\Desktop\Chinese-medical-dialogue-data-master\Data_数据\Surgical_外科\外科5-14000.csv"
# INPUT_FILE = r"C:\Users\27160\Desktop\Chinese-medical-dialogue-data-master\Data_数据\OAGD_妇产科\妇产科6-28000.csv"
INPUT_FILE = r"./样例_内科5000-6000.csv"
MODEL_NAME = "qwen3-embedding:0.6b"
VECTOR_DIM = 1024  # 向量维度(与Milvus Schema一致)
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "Medical_QA_Pairs"
# Set to None to process all data. Set to an integer (e.g. 100) for testing.
LIMIT = None
BATCH_SIZE = 1000  # 批量插入大小

def connect_to_milvus():
    """连接到Milvus服务"""
    try:
        connections.connect(
            alias="default",
            host=MILVUS_HOST,
            port=MILVUS_PORT
        )
        print("成功连接到Milvus服务")
        return True
    except Exception as e:
        print(f"连接Milvus失败: {e}")
        return False

def create_collection():
    """根据提供的Schema创建集合(若不存在)"""
    if utility.has_collection(COLLECTION_NAME):
        print(f"集合 {COLLECTION_NAME} 已存在,将直接使用")
        return Collection(COLLECTION_NAME)
    
    # 定义字段
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="qa_vector", dtype=DataType.BFLOAT16_VECTOR, dim=VECTOR_DIM),
        FieldSchema(name="department", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=256)
    ]
    
    # 定义集合Schema
    schema = CollectionSchema(
        fields=fields,
        description="医疗问答数据集集合"
    )
    
    # 创建集合
    collection = Collection(
        name=COLLECTION_NAME,
        schema=schema,
        using="default"
    )
    print(f"成功创建集合 {COLLECTION_NAME}")
    
    # 创建索引
    index_params = {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 8, "efConstruction": 64}
    }
    result = collection.create_index(
        field_name="qa_vector",
        index_params=index_params
    )
    print("成功为qa_vector字段创建HNSW索引:", result)
    
    return collection

def process_and_upload_data(collection):
    """读取CSV数据,生成embedding并直接上传到Milvus"""
    print(f"读取CSV文件: {INPUT_FILE}...")
    try:
        # 读取CSV文件
        df = pd.read_csv(INPUT_FILE, header=None, names=['department', 'title', 'ask', 'answer'], encoding='utf-8')
    except UnicodeDecodeError:
        print("UTF-8 decode failed, trying gb18030...")
        df = pd.read_csv(INPUT_FILE, header=None, names=['department', 'title', 'ask', 'answer'], encoding='gb18030')

    print(f"成功加载 {len(df)} 行数据")
    
    if LIMIT:
        print(f"处理限制为 {LIMIT} 行")
        # df = df.head(LIMIT)
        df = df.tail(LIMIT)

    
    # 尝试使用tqdm显示进度条
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
    
    print(f"开始生成embeddings和上传数据(批量大小: {BATCH_SIZE})...")
    
    # 准备批量数据的列表
    batch_titles = []
    batch_departments = []
    batch_answers = []
    batch_vectors = []
    
    total_uploaded = 0
    failed_count = 0
    skipped_count = 0  # 跳过的数据计数
    
    iterator = tqdm(df.iterrows(), total=len(df)) if use_tqdm else df.iterrows()
    
    for i, row in iterator:
        # 只处理某些科的数据
        department = str(row['department']) if pd.notna(row['department']) else ""
        if department not in ["眼科"]:
            skipped_count += 1
            continue
        
        # 合并ask和answer作为嵌入输入
        ask_text = str(row['ask']) if pd.notna(row['ask']) else ""
        answer_text = str(row['answer']) if pd.notna(row['answer']) else ""
        combined_text = f"{ask_text} {answer_text}"
        
        # 构建指令嵌入格式
        prompt = f"Instruct: question answering\nQuery: {combined_text}"
        
        try:
            # 生成嵌入
            res = ollama.embeddings(
                model=MODEL_NAME, 
                prompt=prompt
            )
            vec = res['embedding']
            
            # 验证向量维度
            if len(vec) != VECTOR_DIM:
                print(f"\n警告: 第{i}行向量维度不匹配,预期{VECTOR_DIM},实际{len(vec)},跳过此行")
                failed_count += 1
                continue
            
            # 转换为numpy数组
            vec_array = np.array(vec, dtype=np.float32)
            
            # 添加到批量数据
            batch_titles.append(str(row['title']) if pd.notna(row['title']) else "")
            batch_departments.append(department)
            batch_answers.append(answer_text[:256])  # 限制在256字符内
            batch_vectors.append(vec_array)
            
            # 当累积到批量大小时,执行插入
            if len(batch_titles) >= BATCH_SIZE:
                insert_batch(collection, batch_titles, batch_departments, batch_answers, batch_vectors)
                total_uploaded += len(batch_titles)
                print(f"\n已上传 {total_uploaded} 条数据")
                
                # 清空批量数据
                batch_titles = []
                batch_departments = []
                batch_answers = []
                batch_vectors = []
                
        except Exception as e:
            print(f"\n处理第{i}行数据时出错: {e}")
            failed_count += 1
            continue
        
        if not use_tqdm and (i + 1) % 100 == 0:
            print(f"已处理 {i + 1} 行")
    
    # 插入剩余的数据
    if len(batch_titles) > 0:
        insert_batch(collection, batch_titles, batch_departments, batch_answers, batch_vectors)
        total_uploaded += len(batch_titles)
    
    # 刷新集合使数据可见
    collection.flush()
    
    print(f"\n数据导入完成!")
    print(f"成功上传: {total_uploaded} 条")
    print(f"失败: {failed_count} 条")
    print(f"跳过(不符合department条件): {skipped_count} 条")
    print(f"集合总数据量: {collection.num_entities} 条")

def insert_batch(collection, titles, departments, answers, qa_vectors):
    """批量插入数据到Milvus"""
    # 按照Schema字段顺序准备数据
    batch_data = [
        departments,
        titles,
        answers,
        qa_vectors
    ]
    
    try:
        mr = collection.insert(batch_data)
        return True
    except Exception as e:
        print(f"\n批量插入失败: {e}")
        return False

def main():
    # 连接Milvus
    if not connect_to_milvus():
        print("无法连接到Milvus,请确保Milvus服务正在运行")
        return
    
    try:
        # 创建/获取集合
        collection = create_collection()
        
        # 处理数据并上传
        process_and_upload_data(collection)
        
    finally:
        # 关闭连接
        connections.disconnect("default")
        print("已断开与Milvus的连接")

if __name__ == "__main__":
    main()