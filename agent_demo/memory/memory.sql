create database if not exists agent_memory
    character set utf8mb4
    collate utf8mb4_bin;

use agent_memory;
-- 用于初始化新表；CREATE TABLE IF NOT EXISTS 不会修改已有数据库的表结构。
create table if not exists facts(
    id BIGINT unsigned auto_increment primary key,
    user_id varchar(64) not null,         -- 不同的用户
    session_id varchar(64) not null,      -- 来源于不同的会话
    content text not null,         -- 记忆具体的内容
    create_time timestamp not null default current_timestamp comment '创建时间',       -- 创建时间
    kind varchar(64) not null,            -- 内容的类型
    embedding BLOB null,
    embed_model varchar(255) not null,
    is_update boolean not null default false,
    conflict_with BIGINT unsigned null,
    status varchar(16) not null default 'active'
);
