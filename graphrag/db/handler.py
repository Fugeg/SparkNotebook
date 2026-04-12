# graphrag/db/handler.py
import psycopg2
from psycopg2.extras import execute_values

class DatabaseHandler:
    def __init__(self):
        # 对应你 Docker 容器启动时的配置
        self.conn_config = {
            "host": "localhost",
            "database": "ai_notepad",
            "user": "root",
            "password": "123456",
            "port": 5432
        }

    def get_connection(self):
        """获取数据库连接"""
        return psycopg2.connect(**self.conn_config)

    def insert_inspiration(self, content, embedding):
        """
        插入灵感数据 (论文 3.5.1 节直接信息单元)
        """
        query = "INSERT INTO inspirations (content, content_embedding) VALUES (%s, %s) RETURNING id;"
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            # PostgreSQL 的 vector 类型可以直接接受 Python list
            cur.execute(query, (content, embedding))
            new_id = cur.fetchone()[0]
            conn.commit()
            return new_id
        except Exception as e:
            conn.rollback()
            print(f"❌ DB Insert Error (Inspiration): {e}")
            return None
        finally:
            cur.close()
            conn.close()

    def insert_entity(self, table_name, name, description, embedding):
        """
        插入间接信息单元 (人物、地点、事件)
        table_name: 'people', 'places', 'events'
        """
        # 根据表名确定嵌入字段名
        emb_field = "profile_embedding" if table_name == "people" else \
                    "place_embedding" if table_name == "places" else "event_embedding"
        
        query = f"INSERT INTO {table_name} (name, description, {emb_field}) VALUES (%s, %s, %s) RETURNING id;"
        
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            cur.execute(query, (name, description, embedding))
            new_id = cur.fetchone()[0]
            conn.commit()
            return new_id
        except Exception as e:
            conn.rollback()
            print(f"❌ DB Insert Error ({table_name}): {e}")
            return None
        finally:
            cur.close()
            conn.close()

# 单例导出，方便全局调用
db_handler = DatabaseHandler()