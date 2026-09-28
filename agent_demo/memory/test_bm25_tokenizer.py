"""pytest 测试：tokenizer（中英混合分词）+ BM25（Okapi 检索排序）。

依据技能 `python-testing-patterns` 编写：
  - AAA 结构（Arrange / Act / Assert）
  - `@pytest.mark.parametrize` 覆盖等价类，避免重复用例
  - fixture 提供共享语料，保证测试相互隔离
  - 命名遵循 test_<unit>_<scenario>_<expected>
  - 覆盖边界：空串、空语料、OOV 查询

运行：
    cd agent_demo/memory && ../../.venv/bin/python -m pytest test_bm25_tokenizer.py -v
"""
import os
import sys

import pytest

# bm25.py / tokenizer.py 使用同目录平铺导入，故把本目录加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bm25 import BM25  # noqa: E402
from tokenizer import tokenize  # noqa: E402


# ─────────────────────────── fixtures ───────────────────────────

@pytest.fixture
def corpus() -> list[list[str]]:
    """4 篇文档：上海出现于 0/2，北京出现于 1/3，烤鸭仅出现于 3。"""
    return [
        ["我", "爱", "上海"],        # 0
        ["我", "爱", "北京"],        # 1
        ["上海", "上海", "上海"],    # 2，上海出现最多
        ["北京", "烤鸭"],            # 3
    ]


@pytest.fixture
def bm25(corpus: list[list[str]]) -> BM25:
    """基于共享语料构建的检索器。"""
    return BM25(corpus)


# ─────────────────────────── tokenizer ───────────────────────────

@pytest.mark.parametrize(
    "text, expected",
    [
        ("", []),                                  # 空输入
        ("HELLO", ["hello"]),                      # ASCII 统一小写
        ("Hello World", ["hello", "world"]),       # 多个英文词
        ("hello, world!", ["hello", "world"]),     # 标点被丢弃
    ],
)
def test_tokenize_ascii_text_returns_lowercased_words(text, expected):
    # Act
    result = tokenize(text)
    # Assert
    assert result == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("上海", ["上", "海", "上海"]),                                  # 单字 + bigram
        ("我爱上海", ["我", "爱", "上", "海", "我爱", "爱上", "上海"]),    # 2 字 → 2 bigram
    ],
)
def test_tokenize_chinese_text_returns_chars_and_bigrams(text, expected):
    result = tokenize(text)
    assert result == expected


def test_tokenize_mixed_text_splits_by_script_and_number():
    # Arrange / Act
    result = tokenize("Python 3.12 真好用")
    # Assert：数字串按非数字分隔符切分，中文走字 + bigram
    assert result == ["python", "3", "12", "真", "好", "用", "真好", "好用"]


# ─────────────────────────── BM25 初始化 ───────────────────────────

def test_init_records_document_count(bm25, corpus):
    assert bm25.N == len(corpus)


def test_init_average_length_is_mean_of_doc_lengths(bm25, corpus):
    assert bm25.avg == pytest.approx(sum(len(d) for d in corpus) / len(corpus))


def test_init_empty_corpus_uses_default_average_of_one():
    assert BM25([]).avg == 1.0


def test_idf_is_positive_for_every_term(bm25):
    # BM25 的 idf = log(1 + ...) 恒为正，保证打分不出现负值
    assert all(v > 0 for v in bm25.idf.values())


def test_idf_rarer_term_has_higher_weight(bm25):
    # 烤鸭 仅见于 1 篇，北京 见于 2 篇 → 烤鸭 的 idf 必须更大
    assert bm25.idf["烤鸭"] > bm25.idf["北京"]


# ─────────────────────────── BM25 打分 ───────────────────────────

def test_score_all_returns_one_score_per_document(bm25, corpus):
    scores = bm25.score_all(["上海"])
    assert len(scores) == len(corpus)


def test_score_all_unknown_term_yields_all_zeros(bm25, corpus):
    # OOV 词不在词表中，不应对任何文档贡献得分
    assert bm25.score_all(["不存在的词"]) == [0.0] * len(corpus)


def test_score_all_ranks_more_frequent_term_higher(bm25):
    # 文档 2 中"上海"出现 3 次，应高于仅出现 1 次的文档 0
    scores = bm25.score_all(["上海"])
    assert scores[2] > scores[0]


def test_score_all_assigns_zero_to_documents_without_term(bm25):
    scores = bm25.score_all(["上海"])
    assert scores[1] == 0.0
    assert scores[3] == 0.0


def test_score_all_scores_are_never_negative(bm25):
    scores = bm25.score_all(["我", "爱", "上海", "烤鸭", "OOV"])
    assert all(s >= 0.0 for s in scores)


def test_score_all_empty_corpus_returns_empty_list():
    assert BM25([]).score_all(["任何词"]) == []


def test_score_all_top1_for_shanghai_is_most_frequent_doc(bm25):
    # 端到端：查询"上海"，排名第一应为文档 2
    scores = bm25.score_all(["上海"])
    assert scores.index(max(scores)) == 2
