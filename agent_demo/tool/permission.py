import threading

ALLOW, DENY, ASK = "allow","deny","ask"
PERMISSION_RULES = {
    "read_file": ALLOW,
    "calculate": ALLOW,
    "write_file": ASK,  # 有副作用，每次问一下
}
class PermissionPolicy:
    """
    一维权限策略，通过tool_name判断是否有权限
    检查函数可以多留一个参数 args ，用于后续看具体参数来升级二维权限策略
    """
    def __init__(self,rules:dict|None = None):
        self.rules = dict(PERMISSION_RULES if rules is None else rules)
        # 定义一个列表存放本轮一直运行执行的工具
        self.approve = set()

    def check(self,tool_name:str,args:dict | None = None):
        action = self.rules.get(tool_name,ASK)
        if action == DENY:
            return DENY
        if action == ALLOW:
            return ALLOW
        if action == ASK:
            if tool_name in self.approve:
                return ALLOW
            return ASK
        return DENY

    def approve_tool(self,tool_name:str):
        self.approve.add(tool_name)

class PermissionGate:
    def __init__(self,policy:PermissionPolicy | None = None):
        self.policy = PermissionPolicy() if policy is None else policy
        self._ask_lock = threading.Lock()
    def guard(self,tool_name:str,args:dict | None = None):
        """返回None表示放行，返回str表示拒绝原因"""
        action = self.policy.check(tool_name,args)
        if action == DENY:
            return f"权限拒绝：规则禁止调用{tool_name}"
        elif action == ASK:
            with self._ask_lock:
                choice = input(
                    f"允许执行{tool_name}({args})?\n[once/always/reject]\n"
                )
                if choice == "always":
                    self.policy.approve_tool(tool_name)
                    return None
                elif choice == "reject":
                    return f"权限拒绝：用户拒绝调用{tool_name}"
                elif choice == "once":
                    return None
                else:
                    return f"输入不合法，因此拒绝调用"
        elif action == ALLOW:
            return None
