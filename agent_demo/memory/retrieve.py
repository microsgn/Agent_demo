"""按用户检索长期记忆，使用 BM25、余弦相似度和 RRF 排名融合。"""
from contextlib import closing

import numpy as np
import pymysql

from models.DB_Config import DB_CONFIG
from . import embed
from .bm25 import BM25
from .tokenizer import tokenize

DEFAULT_LIMIT = 5
MIN_VECTOR_SIM = 0.5
RRF_K = 60


def retrieve(
    text: str,
    user_id: str,
    limit: int = DEFAULT_LIMIT,
) -> list[dict]:
    """返回按相关性排序的候选记忆，保留数据库 ID 供冲突判断使用。

    查询所有属于该用户的事实，不按 session_id 限制长期记忆。
    当前表没有模型版本字段，统一通过当前 embed 的缓存获取向量，
    不使用数据库里来源不明的旧 embedding BLOB。
    """
    if not isinstance(text, str):
        raise TypeError("text 必须是字符串")
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("user_id 必须是非空字符串")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("limit 必须是非负整数")
    if not text.strip() or limit == 0:
        return []

    sql = """
        SELECT id, content, kind
        FROM facts
        WHERE user_id = %s AND embed_model = %s AND status = 'active'
        ORDER BY id
    """
    with closing(pymysql.connect(**DB_CONFIG)) as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, (user_id,embed.EMBED_MODEL))
            rows = cursor.fetchall()
    if not rows:
        return []

    contents = [row["content"] for row in rows]
    if any(not isinstance(content, str) or not content.strip() for content in contents):
        raise ValueError("数据库中存在内容为空或类型错误的记忆")

    # 两路使用同一份 rows；下标始终对应同一条数据库记录。
    bm25_scores = BM25([tokenize(content) for content in contents]).score_all(
        tokenize(text)
    )

    # 文本向量均使用当前模型；embed 内部负责缓存、去重和分批。
    vectors = np.asarray(embed.embed([text, *contents]), dtype=np.float64)
    expected_shape = (len(rows) + 1, embed.DIM)
    if vectors.shape != expected_shape or not np.isfinite(vectors).all():
        raise ValueError("embedding 数量、维度不匹配，或包含非有限值")
    norms = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError("embedding 的范数必须是有限正数")
    unit_vectors = vectors / norms[:, None]
    vector_scores = np.clip(unit_vectors[1:] @ unit_vectors[0], -1.0, 1.0)

    # 独立召回：向量检索覆盖全部记忆，不局限于 BM25 的命中结果。
    bm25_order = sorted(
        (i for i, score in enumerate(bm25_scores) if score > 0),
        key=lambda i: bm25_scores[i],
        reverse=True,
    )
    vector_order = sorted(
        (i for i, score in enumerate(vector_scores) if score >= MIN_VECTOR_SIM),
        key=lambda i: vector_scores[i],
        reverse=True,
    )

    # RRF 只融合名次，避免直接相加两种量纲不同的原始分数。
    fused: dict[int, float] = {}
    for order in (bm25_order, vector_order):
        for rank, i in enumerate(order, start=1):
            fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank)

    qualified = [i for i in fused if vector_scores[i] >= MIN_VECTOR_SIM]
    top_indices = sorted(
        qualified, key=lambda i: (-fused[i], rows[i]["id"])
    )[:limit]
    return [
        {
            "id": rows[i]["id"],
            "content": rows[i]["content"],
            "kind": rows[i]["kind"],
        }
        for i in top_indices
    ]
