from pydantic import BaseModel,Field
from typing import Literal

class MemoryItem(BaseModel):
    should_store : bool = Field(description = "是否应该存入长期记忆的判断")
    kind : Literal["identity","address","preference","experience","default"] = Field(
        description = "记忆类别"
    )
    importance : float = Field(description= "存入数据库的重要程度,范围为0.0-1.0,数值越大越有必要存入长期记忆")
    confidence : float = Field(description="判断的置信程度，范围为0.0-1.0,置信程度越高，判断结果越肯定")
    content : str = Field(description = "原文内容，必须与输入一致")
    is_update : bool = Field(description="是否是信息的更新")
    conflict_with : int | None = Field(description="冲突信息的id，如果不冲突为None(Null)")

class MemoryBatch(BaseModel):
    items : list[MemoryItem] = Field(description= "分类结果列表")

class FactsResponse(BaseModel):
    index : int=Field(description="对应的输入消息")
    response : list=Field(description="检索的结果")