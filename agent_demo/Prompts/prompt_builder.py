from .call_llm import call_llm,CompactionError
from .tokens import remain_tokens,estimate_tokens,need_compaction
from .sys_prompt import System_prompt
class context:
    def __init__(self,N:int,system_prompt:System_prompt):
        self.system_list = []
        self.history_list = []
        self.recent_list = []
        self.compaction_list = []
        self.N = N
        self.context = []
        self.system_prompt = system_prompt

    def build_system(self):
        self.system_list.append({"role":"system","content":self.system_prompt.assemble()})
    def build_recent(self,history:list[dict]):
        if len(self.recent_list) < self.N:
            self.recent_list.append(history)
        else:
            msg = self.recent_list.pop(0)
            self.history_list.extend(msg)
            self.recent_list.append(history)

    def compaction_list(self,compact_context:str):
        self.compaction_list.append({"role":"assistant","content":compact_context})
        self.history_list.clear()

    def compaction(self)->bool:
        if not self.history_list:
            return False
        try:
            summary = call_llm(self.history_list.copy())
        except CompactionError as exc:
            print(f"压缩失败，历史已保留:{exc}")
            return False
        self.compaction_list.append({
            "role": "assistant",
            "content": f"【历史会话摘要】\n{summary}",
        })
        self.history_list.clear()
        return True

    def build_context(self,references: list[dict],current_turn:list[dict])->list[dict]:
        self.context.clear()
        self.context.extend(self.system_list)
        self.context.extend(self.compaction_list)
        self.context.extend(self.history_list)
        for msg in self.recent_list:
            self.context.extend(msg)
        self.context.extend(references)
        self.context.extend(current_turn)
        return self.context

    def prepare_context(self,references:list[dict],current_turn:list[dict],tools:list[dict]|None = None)->list[dict]:
        self.system_list.clear()
        self.build_system()
        messages = self.build_context(references,current_turn)
        input_tokens = estimate_tokens(messages,tools)
        remaining = remain_tokens(input_tokens)

        if need_compaction(remaining):
            if self.compaction():
                messages = self.build_context(references,current_turn)
                input_tokens = estimate_tokens(messages, tools)
                remaining = remain_tokens(input_tokens)
            else:
                print("本次未完成压缩，保留原上下文。")
        if remaining < 0:
            raise RuntimeError(
                f"上下文预计超出预算 {-remaining} tokens，"
                "本次停止调用模型。"
            )
        return messages