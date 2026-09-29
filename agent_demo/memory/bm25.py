"""标准 BM25（Okapi）。k1 控制词频饱和，b 控制长度归一化。"""
import math
from collections import Counter

class BM25:
    def __init__(self,docs_tokens:list[list[str]],k1:float = 1.5,b:float = 0.75):
        self.k1 = k1
        self.b = b
        self.N = len(docs_tokens)
        self.length = [len(d) for d in docs_tokens]
        self.avg = (sum(self.length)/self.N) if self.N > 0 else 1.0

        self.tf = [Counter(d) for d in docs_tokens]
        self.df = Counter()
        for c in self.tf:
            self.df.update(c.keys())

        self.idf = {w:math.log(1+(self.N - n + 0.5)/(n + 0.5))for w,n in self.df.items()}

        self.postings:dict[str,list[tuple[int,int]]] = {}
        for i,c in enumerate(self.tf):
            for w,f in c.items():
                self.postings.setdefault(w,[]).append((i,f))

    def score_all(self,query_tokens:list[str])->list[float]:
        """计算BM25检索的得分"""
        out = [0.0] * self.N
        for w in set(query_tokens):
            idf = self.idf.get(w)
            if idf is None:
                continue
            for i,f in self.postings[w]:
                denom = f + self.k1*(1-self.b + self.b * self.len[i] / self.avg)
                out[i] += idf * f *(self.k1 + 1) / denom
        return out

if __name__=="__main__":
    docs = [
        ["我", "爱", "上海"],  # 0
        ["我", "爱", "北京"],  # 1
        ["上海", "上海", "上海"],  # 2  ← 上海 出现最多
        ["北京", "烤鸭"],  # 3
    ]

    bm = BM25(docs)

    # 查询：上海
    scores = bm.score_all(["上海"])
    print("query = 上海")
    for i, s in enumerate(scores):
        print(f"  文档{i} {docs[i]}  →  score = {s:.4f}")

    # 排序取 top2
    rank = sorted(range(len(docs)), key=lambda i: -scores[i])[:2]
    print("\nTop2 最相关：")
    for i in rank:
        print(f"  文档{i}: {docs[i]}  ({scores[i]:.4f})")