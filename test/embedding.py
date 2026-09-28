from agent_demo.memory.embed import embed_one
from agent_demo.memory.store import  Store

def cos(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    na = sum(x*x for x in a) ** 0.5
    nb = sum(x*x for x in b) ** 0.5
    return dot / (na * nb)

base = embed_one("用户的居住地是上海")
tests = [
    ("用户常住上海",        "应该最高"),
    ("用户居住在上海",      "应该很高"),
    ("用户居住在北京",      "★ 关键：也应该很高！"),
    ("用户不喜欢吃辣",      "应该低"),
    ("今天股市大涨",        "应该最低"),
]
for text, note in tests:
    print(f"{cos(base, embed_one(text)):.4f}  {text}   ({note})")