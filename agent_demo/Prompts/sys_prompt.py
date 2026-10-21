import os,sys
from models.DB_Config import DB_CONFIG
import pymysql
MAX_TEXT_lENGTH = 2000
from contextlib import contextmanager

@contextmanager
def get_conn():
    conn = pymysql.connect(**DB_CONFIG)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def _detect_git(cwd: str) -> bool:
    path = os.path.abspath(cwd)
    while True:
        git_path = os.path.join(path, ".git")
        if os.path.exists(git_path):        # 目录或文件都算
            return True
        parent = os.path.dirname(path)
        if parent == path:                  # 到根目录了
            return False
        path = parent


class System_prompt:
    """采取OpenCode设计模式"""
    def __init__(self,user_id:str = "default"):
        self.custom: str | None = None
        self.user_id = user_id

    def header(self,agent_name:str = "build")->str:
        """提示词身份层，用于身份说明，同样可以用于区分多Agent"""
        agents = {
            "build":"""
                你是一个通用型Agent，主要用来进行编程和人工智能以及数理学习
                你可以通过调用相关工具，实现用户的特定需求
                你的目标是解答用户的疑问
            """
        }
        return agents.get(agent_name,"build")

    def provider(self)->str:
        """提示词静态层，主要说明一些固定的规则"""
        try:
            with open("/Users/microsgn/Agent/agent_demo/Prompts/静态规则.txt","r",encoding="utf-8") as f:
                content = f.read()
        except FileNotFoundError:
            raise f"文件读取失败"
        static = content
        return static

    def environment(self)->str:
        """提示词环境层，告诉LLM当前工作空间等动态信息"""
        cwd = os.getcwd()
        is_git = _detect_git(cwd)
        return "\n".join([
            f" 工作目录：{cwd} "
            f" 是否为git仓库:{"是"if is_git is True else "不是"}"
            f" 工作平台:{sys.platform}"
        ])

    def save_custom_prompt(self,content: str):
        """
        用户提示词，由LLM帮用户写入数据库中,并由后续过程查出来粘贴到系统提示词中，内容要少于2000字符
        :param content: LLM根据用户提示所输出的用户偏好
        """
        if not isinstance(content, str):
            return "错误信息：提示词必须为字符串"
        if not content.strip():
            return "错误信息：提示词不能为空或只包含空白"
        sql = f"""
        insert into user_custom (user_id,content) values(%s,%s)
        on duplicate key update content = values(content)
    """
        try:
            if len(content) > MAX_TEXT_lENGTH:
                raise ValueError("提示词过长")
            with get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, (self.user_id, content))
                return True
        except ValueError as e:
            return f"错误信息：{e}"

    def load_custom(self) -> str:
        sql = f"""
            select content from user_custom where user_id = %s
        """
        with get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (self.user_id,))
            rows = cursor.fetchall()
        if not rows:
            return ""
        else:
            return rows[0]['content']

    def assemble(self):
            return "\n\n".join(
                [
                    self.header(),
                    self.provider(),
                    self.environment(),
                    self.load_custom(),
                ]
            )
