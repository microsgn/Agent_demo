CREATE DATABASE IF NOT EXISTS agent_memory
  DEFAULT CHARACTER SET utf8mb4          -- 必须！否则中文、emoji 会乱码
  COLLATE utf8mb4_unicode_ci;

USE agent_memory;

-- ── L2 短期记忆：原始对话轮次 ────────────────────────
CREATE TABLE turns (
    id         BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(64)  NOT NULL,     -- 你的 session 标识
    role       ENUM('user','assistant') NOT NULL,
    content    MEDIUMTEXT   NOT NULL,     -- 单轮可能很长，用 MEDIUMTEXT
    created_at DOUBLE       NOT NULL,     -- Unix 秒（带小数）
    INDEX idx_session (session_id, id)    -- 按 session 查最近 N 轮
) ENGINE=InnoDB;

-- ── L2 压缩层：会话摘要 ──────────────────────────────
CREATE TABLE summaries (
    id          BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    session_id  VARCHAR(64) NOT NULL,
    covers_from BIGINT UNSIGNED NOT NULL,  -- 覆盖的 turn id 起点
    covers_to   BIGINT UNSIGNED NOT NULL,
    content     TEXT        NOT NULL,
    created_at  DOUBLE      NOT NULL,
    INDEX idx_session (session_id, covers_from)
) ENGINE=InnoDB;

-- ── L3 长期记忆：跨会话事实 ──────────────────────────
CREATE TABLE facts (
    id            BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    user_id       VARCHAR(64)  NOT NULL DEFAULT 'default',
    content       VARCHAR(1000) NOT NULL,  -- 原子事实，一句话，够用
    kind          VARCHAR(32)  NOT NULL,   -- preference/constraint/identity/goal/skill
    source        VARCHAR(32)  NOT NULL,   -- user_explicit/agent_inferred/skill_success
    scene         VARCHAR(64),             -- 来源 session，仅供追溯
    confidence    FLOAT        NOT NULL DEFAULT 0.9,
    created_at    DOUBLE       NOT NULL,
    valid_until   DOUBLE       NULL,       -- NULL = 有效；被推翻时写时间戳
    superseded_by BIGINT UNSIGNED NULL,
    use_count     INT UNSIGNED NOT NULL DEFAULT 0,
    last_used     DOUBLE       NULL,
    embed         BLOB         NULL,       -- 向量：float32 打包成二进制
    INDEX idx_active (user_id, valid_until),   -- 最常用：查某用户的有效记忆
    INDEX idx_kind   (user_id, kind, valid_until)
) ENGINE=InnoDB;

-- ── L4 程序性记忆：skill / 工具用法 ──────────────────
CREATE TABLE skills (
    id          BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    user_id     VARCHAR(64) NOT NULL DEFAULT 'default',
    name        VARCHAR(128) NOT NULL,
    description TEXT        NOT NULL,
    usage_notes       TEXT        NULL,          -- 怎么调用、参数、注意事项
    created_at  DOUBLE      NOT NULL,
    use_count   INT UNSIGNED NOT NULL DEFAULT 0,
    UNIQUE KEY uk_name (user_id, name)
) ENGINE=InnoDB;
