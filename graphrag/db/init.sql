-- GraphRAG AI Notebook 数据库初始化脚本
-- 数据库: sparknotebook
-- 创建时间: 2026-04-14

-- 启用 vector 扩展 (用于向量相似度搜索)
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================
-- 1. 用户表
-- ============================================
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) NOT NULL UNIQUE,
    email VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- 2. 原始输入表 (对话历史)
-- ============================================
CREATE TABLE IF NOT EXISTS raw_inputs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    main_content TEXT NOT NULL,
    audio_link TEXT,
    input_method VARCHAR(50) DEFAULT 'text',
    response_content TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- 3. 灵感表 (直接信息单元)
-- ============================================
CREATE TABLE IF NOT EXISTS inspirations (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE DEFAULT 1,
    content TEXT NOT NULL,
    content_embedding vector(1024),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- 4. 提醒表
-- ============================================
CREATE TABLE IF NOT EXISTS reminders (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE DEFAULT 1,
    content TEXT NOT NULL,
    content_embedding vector(1024),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- 5. 经验表
-- ============================================
CREATE TABLE IF NOT EXISTS experiences (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE DEFAULT 1,
    content TEXT NOT NULL,
    content_embedding vector(1024),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- 6. 杂思表
-- ============================================
CREATE TABLE IF NOT EXISTS miscellaneous_thoughts (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE DEFAULT 1,
    content TEXT NOT NULL,
    content_embedding vector(1024),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- 7. 人物表 (间接信息单元)
-- ============================================
CREATE TABLE IF NOT EXISTS people (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE DEFAULT 1,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    profile_embedding vector(1024),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- 8. 地点表
-- ============================================
CREATE TABLE IF NOT EXISTS places (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE DEFAULT 1,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    place_embedding vector(1024),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- 9. 事件表
-- ============================================
CREATE TABLE IF NOT EXISTS events (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE DEFAULT 1,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    event_embedding vector(1024),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- 10. 连接表 (关系)
-- ============================================
CREATE TABLE IF NOT EXISTS connections (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE DEFAULT 1,
    content TEXT NOT NULL,
    content_embedding vector(1024),
    connection_type VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- 11. 边表 (图结构)
-- ============================================
CREATE TABLE IF NOT EXISTS edges (
    id SERIAL PRIMARY KEY,
    connection_id INTEGER REFERENCES connections(id) ON DELETE CASCADE,
    node_type VARCHAR(100) NOT NULL,
    node_id INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- 12. 实体别名表 (新增 - 用于实体归一化)
-- ============================================
CREATE TABLE IF NOT EXISTS entity_aliases (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE DEFAULT 1,
    canonical_name VARCHAR(255) NOT NULL,
    alias_name VARCHAR(255) NOT NULL,
    entity_type VARCHAR(100) NOT NULL,
    entity_id INTEGER,
    confidence FLOAT DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(canonical_name, alias_name, entity_type)
);

-- ============================================
-- 13. 并查集状态表 (新增 - 用于持久化 DSU)
-- ============================================
CREATE TABLE IF NOT EXISTS entity_resolver_state (
    id SERIAL PRIMARY KEY,
    entity_name VARCHAR(255) NOT NULL UNIQUE,
    parent_name VARCHAR(255),
    rank_value INTEGER DEFAULT 0,
    frequency INTEGER DEFAULT 1,
    canonical_name VARCHAR(255),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- 创建索引 (优化查询性能)
-- ============================================

-- 用户表索引
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);

-- 原始输入表索引
CREATE INDEX IF NOT EXISTS idx_raw_inputs_user_id ON raw_inputs(user_id);
CREATE INDEX IF NOT EXISTS idx_raw_inputs_created_at ON raw_inputs(created_at);

-- 向量相似度搜索索引 (使用 ivfflat 或 hnsw)
CREATE INDEX IF NOT EXISTS idx_inspirations_embedding ON inspirations USING ivfflat (content_embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_reminders_embedding ON reminders USING ivfflat (content_embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_experiences_embedding ON experiences USING ivfflat (content_embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_miscellaneous_embedding ON miscellaneous_thoughts USING ivfflat (content_embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_people_embedding ON people USING ivfflat (profile_embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_places_embedding ON places USING ivfflat (place_embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_events_embedding ON events USING ivfflat (event_embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_connections_embedding ON connections USING ivfflat (content_embedding vector_cosine_ops);

-- 边表索引
CREATE INDEX IF NOT EXISTS idx_edges_connection_id ON edges(connection_id);
CREATE INDEX IF NOT EXISTS idx_edges_node ON edges(node_type, node_id);

-- 实体别名表索引 (新增)
CREATE INDEX IF NOT EXISTS idx_aliases_canonical ON entity_aliases(canonical_name);
CREATE INDEX IF NOT EXISTS idx_aliases_alias ON entity_aliases(alias_name);
CREATE INDEX IF NOT EXISTS idx_aliases_type ON entity_aliases(entity_type);
CREATE INDEX IF NOT EXISTS idx_resolver_state_parent ON entity_resolver_state(parent_name);
CREATE INDEX IF NOT EXISTS idx_resolver_state_canonical ON entity_resolver_state(canonical_name);

-- ============================================
-- 插入默认用户
-- ============================================
INSERT INTO users (id, username, email) VALUES (1, 'default_user', 'default@example.com') ON CONFLICT DO NOTHING;

-- ============================================
-- 完成
-- ============================================
SELECT '数据库初始化完成！' AS status;
