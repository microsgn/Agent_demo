import os
from dotenv import load_dotenv
load_dotenv()
import pymysql
DB_CONFIG = dict(
    host = "127.0.0.1",
    port = 3306,
    user = "root",
    password =os.getenv("PASSWORD"),
    database = "agent_memory",
    charset = "utf8mb4",
    cursorclass = pymysql.cursors.DictCursor,
)
DB_field = [
    "id",
    "user_id",
    "session_id",
    "content",
    "created_time"
    "kind",
    "embedding"
    "is_update"
    "conflict_with"
]
