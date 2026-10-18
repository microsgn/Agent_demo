from openai import OpenAI
import os
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

load_dotenv()

PROMPT_PATH = Path(__file__).resolve().parent / "压缩.md"
SUMMARY_MODEL = "deepseek-flash"
MAX_SUMMARY_TOKENS = 4096


class CompactionError(RuntimeError):
    """压缩调用失败，或者返回的摘要不可用。"""


def call_llm(
    history: list[dict],
    max_summary_tokens: int = MAX_SUMMARY_TOKENS,
) -> str:
    if not history:
        raise CompactionError("没有需要压缩的内容")

    if max_summary_tokens <= 0:
        raise CompactionError("摘要输出上限必须大于 0")

    # 1. 读取提示词
    try:
        system_prompt = PROMPT_PATH.read_text(
            encoding="utf-8"
        ).strip()
    except OSError as exc:
        raise CompactionError("无法读取压缩提示词") from exc

    if not system_prompt:
        raise CompactionError("压缩提示词为空")

    # 2. 保留原消息的角色和内容，作为待总结资料
    try:
        source_text = json.dumps(
            history,
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CompactionError(
            "待压缩内容无法转换为 JSON 文本"
        ) from exc

    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_URL")

    if not api_key or not base_url:
        raise CompactionError(
            "缺少 DEEPSEEK_API_KEY 或 DEEPSEEK_URL 配置"
        )

    # 3. 调用模型；连接使用完毕后关闭
    try:
        with OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=30.0,
            max_retries=1,
        ) as client:
            response = client.chat.completions.create(
                model=SUMMARY_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": source_text,
                    },
                ],
                max_tokens=max_summary_tokens,
                stream=False,
                extra_body={
                    "thinking": {"type": "disabled"}
                },
            )
    except OpenAIError as exc:
        raise CompactionError(
            f"压缩模型调用失败：{type(exc).__name__}"
        ) from exc

    # 4. 检查回复是否完整
    if not response.choices:
        raise CompactionError("压缩模型没有返回候选回复")

    choice = response.choices[0]

    if choice.finish_reason != "stop":
        raise CompactionError(
            "摘要未正常结束："
            f"finish_reason={choice.finish_reason}"
        )

    if choice.message.refusal:
        raise CompactionError("模型拒绝生成摘要")

    # 5. 检查文本是否可用
    content = choice.message.content

    if not isinstance(content, str) or not content.strip():
        raise CompactionError("模型返回了空摘要")

    summary = content.strip()

    if not summary.startswith("# 会话摘要"):
        raise CompactionError(
            "摘要不符合约定的 Markdown 格式"
        )

    has_body = any(
        line.strip()
        and not line.lstrip().startswith("#")
        for line in summary.splitlines()
    )
    if not has_body:
        raise CompactionError("摘要只有标题，没有正文")

    return summary