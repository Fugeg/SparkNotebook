import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

class Database:
    def __init__(self):
        self.connection_string = os.getenv('DATABASE_URL', 'postgresql://root:123456@localhost:5432/ai_notepad')
        self.connection = None
        self.cursor = None
    
    def connect(self):
        try:
            self.connection = psycopg2.connect(self.connection_string)
            self.cursor = self.connection.cursor()
            return True
        except Exception as e:
            print(f"数据库连接失败: {e}")
            return False
    
    def disconnect(self):
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
    
    def table_exists(self, table_name):
        try:
            self.cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = %s
                )
            """, (table_name,))
            return self.cursor.fetchone()[0]
        except Exception as e:
            print(f"检查表 {table_name} 是否存在失败: {e}")
            return False
    
    def initialize(self):
        if not self.connect():
            return False
        try:
            print("数据库表已存在，跳过创建")
            return True
        except Exception as e:
            print(f"数据库初始化失败: {e}")
            return False
        finally:
            self.disconnect()
    
    def create_user(self, username, email=None):
        """创建新用户"""
        if not self.connect():
            return None
        
        try:
            # 如果 email 为空，设为 NULL 而不是空字符串
            email_value = email if email and email.strip() else None
            
            query = """
            INSERT INTO users (username, email)
            VALUES (%s, %s)
            RETURNING id
            """
            self.cursor.execute(query, (username, email_value))
            user_id = self.cursor.fetchone()[0]
            self.connection.commit()
            return user_id
        except Exception as e:
            print(f"创建用户失败：{e}")
            self.connection.rollback()
            return None
        finally:
            self.disconnect()
    
    def get_user_by_username(self, username):
        """根据用户名获取用户"""
        if not self.connect():
            return None
        
        try:
            query = "SELECT id, username, email FROM users WHERE username = %s"
            self.cursor.execute(query, (username,))
            row = self.cursor.fetchone()
            if row:
                return {
                    'id': row[0],
                    'username': row[1],
                    'email': row[2]
                }
            return None
        except Exception as e:
            print(f"获取用户失败: {e}")
            return None
        finally:
            self.disconnect()
    
    def list_users(self):
        """列出所有用户"""
        if not self.connect():
            return []
        
        try:
            query = "SELECT id, username, email FROM users"
            self.cursor.execute(query)
            users = []
            for row in self.cursor.fetchall():
                users.append({
                    'id': row[0],
                    'username': row[1],
                    'email': row[2]
                })
            return users
        except Exception as e:
            print(f"列出用户失败: {e}")
            return []
        finally:
            self.disconnect()
    
    def insert_raw_input(self, main_content, audio_link=None, input_method='text', user_id=1, response_content=None):
        """插入原始输入到raw_inputs表"""
        if not self.connect():
            return None
        
        try:
            query = """
            INSERT INTO raw_inputs (user_id, main_content, audio_link, input_method, response_content)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """
            self.cursor.execute(query, (user_id, main_content, audio_link, input_method, response_content))
            raw_input_id = self.cursor.fetchone()[0]
            self.connection.commit()
            return raw_input_id
        except Exception as e:
            print(f"插入原始输入失败: {e}")
            self.connection.rollback()
            return None
        finally:
            self.disconnect()
    
    def update_raw_input_response(self, raw_input_id, response_content):
        """更新原始输入的回复内容"""
        if not self.connect():
            return False
        
        try:
            query = """
            UPDATE raw_inputs
            SET response_content = %s
            WHERE id = %s
            """
            self.cursor.execute(query, (response_content, raw_input_id))
            self.connection.commit()
            return True
        except Exception as e:
            print(f"更新回复内容失败: {e}")
            self.connection.rollback()
            return False
        finally:
            self.disconnect()
    
    def get_chat_history(self, user_id, limit=20):
        """获取用户的对话历史"""
        if not self.connect():
            return []
        
        try:
            query = """
            SELECT id, main_content, audio_link, input_method, response_content, created_at
            FROM raw_inputs
            WHERE user_id = %s AND response_content IS NOT NULL
            ORDER BY created_at DESC
            LIMIT %s
            """
            self.cursor.execute(query, (user_id, limit))
            rows = self.cursor.fetchall()
            
            history = []
            for row in reversed(rows):
                history.append({
                    'id': row[0],
                    'user_input': row[1],
                    'audio_link': row[2],
                    'input_method': row[3],
                    'response': row[4],
                    'created_at': row[5]
                })
            return history
        except Exception as e:
            print(f"获取对话历史失败: {e}")
            return []
        finally:
            self.disconnect()
    
    def insert_node(self, table_name, content, metadata=None, embedding=None, user_id=1):
        if not self.connect():
            return None
        
        # 所有表都使用1024维
        table_config = {
            'inspirations': ('content', 'content_embedding', 1024),
            'reminders': ('content', 'content_embedding', 1024),
            'experiences': ('content', 'content_embedding', 1024),
            'miscellaneous_thoughts': ('content', 'content_embedding', 1024),
            'people': ('name', 'profile_embedding', 1024),
            'events': ('title', 'event_embedding', 1024),
            'places': ('name', 'place_embedding', 1024),
            'connections': ('content', 'content_embedding', 1024)
        }
        
        config = table_config.get(table_name, {'content_col': 'content', 'embedding_col': 'content_embedding', 'dim': 1024})
        
        if embedding:
            if len(embedding) != config[2]:
                print(f"警告: 嵌入维度不匹配，期望{config[2]}维，实际{len(embedding)}维")
                if len(embedding) > config[2]:
                    embedding = embedding[:config[2]]
                else:
                    embedding = embedding + [0.0] * (config[2] - len(embedding))
        
        try:
            content_col = config[0]
            embedding_col = config[1]
            
            query = f"INSERT INTO {table_name} (user_id, {content_col}, {embedding_col}) VALUES (%s, %s, %s) RETURNING id"
            
            self.cursor.execute(query, (user_id, content, embedding))
            node_id = self.cursor.fetchone()[0]
            self.connection.commit()
            return node_id
        except Exception as e:
            print(f"插入节点失败: {e}")
            self.connection.rollback()
            return None
        finally:
            self.disconnect()
    
    def insert_edge(self, source_type, source_id, target_type, target_id, relationship_type):
        if not self.connect():
            return None
        
        try:
            connection_content = f"{relationship_type}: {source_type}(id={source_id}) <-> {target_type}(id={target_id})"
            empty_embedding = [0.0] * 1024
            
            query = """
            INSERT INTO connections (content, content_embedding, connection_type)
            VALUES (%s, %s, %s)
            RETURNING id
            """
            self.cursor.execute(query, (connection_content, empty_embedding, relationship_type))
            connection_id = self.cursor.fetchone()[0]
            
            self.cursor.execute("""
            INSERT INTO edges (connection_id, node_type, node_id)
            VALUES (%s, %s, %s)
            """, (connection_id, source_type, source_id))
            
            self.cursor.execute("""
            INSERT INTO edges (connection_id, node_type, node_id)
            VALUES (%s, %s, %s)
            """, (connection_id, target_type, target_id))
            
            self.connection.commit()
            return connection_id
        except Exception as e:
            print(f"插入边失败: {e}")
            self.connection.rollback()
            return None
        finally:
            self.disconnect()
    
    def search_similar_nodes(self, embedding, top_k=5, user_id=1):
        if not self.connect():
            return []
        
        try:
            table_configs = {
                'inspirations': ('content', 'content_embedding', 1024),
                'reminders': ('content', 'content_embedding', 1024),
                'experiences': ('content', 'content_embedding', 1024),
                'miscellaneous_thoughts': ('content', 'content_embedding', 1024),
                'people': ('name', 'profile_embedding', 1024),
                'events': ('title', 'event_embedding', 1024),
                'places': ('name', 'place_embedding', 1024),
            }
            
            results = []
            for table_name, (content_col, embedding_col, dim) in table_configs.items():
                if not self.table_exists(table_name):
                    continue
                
                search_embedding = embedding[:dim] if len(embedding) > dim else embedding + [0.0] * (dim - len(embedding))
                embedding_str = '[' + ','.join(map(str, search_embedding)) + ']'
                
                try:
                    query = f"SELECT id, {content_col}, {embedding_col} <-> %s::vector as distance FROM {table_name} WHERE user_id = %s ORDER BY distance LIMIT %s"
                    
                    self.cursor.execute(query, (embedding_str, user_id, top_k))
                    for row in self.cursor.fetchall():
                        results.append({
                            'id': row[0],
                            'type': table_name,
                            'content': row[1],
                            'metadata': {},
                            'distance': row[2]
                        })
                except Exception as e:
                    print(f"搜索表 {table_name} 失败: {e}")
            
            results.sort(key=lambda x: x['distance'])
            return results[:top_k]
        except Exception as e:
            print(f"搜索相似节点失败: {e}")
            return []
        finally:
            self.disconnect()
    
    def get_node_neighbors(self, node_type, node_id, user_id=1):
        if not self.connect():
            return []
        
        try:
            self.cursor.execute("""
            SELECT e.connection_id, c.content, c.connection_type
            FROM edges e
            JOIN connections c ON e.connection_id = c.id
            WHERE e.node_type = %s AND e.node_id = %s AND c.user_id = %s
            """, (node_type, node_id, user_id))
            
            neighbors = []
            for row in self.cursor.fetchall():
                connection_id = row[0]
                
                self.cursor.execute("""
                SELECT node_type, node_id
                FROM edges
                WHERE connection_id = %s AND NOT (node_type = %s AND node_id = %s)
                """, (connection_id, node_type, node_id))
                
                for neighbor_row in self.cursor.fetchall():
                    neighbors.append({
                        'type': neighbor_row[0],
                        'id': neighbor_row[1],
                        'relationship': row[2]
                    })
            
            return neighbors
        except Exception as e:
            print(f"获取节点邻居失败: {e}")
            return []
        finally:
            self.disconnect()
    
    def get_node_by_id(self, node_type, node_id):
        if not self.connect():
            return None
        
        try:
            table_configs = {
                'inspirations': ('content', 'content_embedding'),
                'reminders': ('content', 'content_embedding'),
                'experiences': ('content', 'content_embedding'),
                'miscellaneous_thoughts': ('content', 'content_embedding'),
                'people': ('name', 'profile_embedding'),
                'events': ('title', 'event_embedding'),
                'places': ('name', 'place_embedding'),
                'connections': ('content', 'content_embedding')
            }
            
            config = table_configs.get(node_type)
            if not config:
                return None
            
            content_col = config[0]
            
            if not self.table_exists(node_type):
                return None
            
            query = f"SELECT id, {content_col} FROM {node_type} WHERE id = %s"
            
            self.cursor.execute(query, (node_id,))
            row = self.cursor.fetchone()
            if row:
                return {
                    'id': row[0],
                    'content': row[1],
                    'metadata': {}
                }
            return None
        except Exception as e:
            print(f"获取节点失败: {e}")
            return None
        finally:
            self.disconnect()