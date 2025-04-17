import time
from google import genai
from google.genai.types import GenerateContentConfig

MODEL = "gemini-2.0-flash-001"


def inference_chat(
    chat_history,
    model: str            = MODEL,
    api_url: str          = None,   # 位置参数兼容，但不用于初始化
    api_key: str          = None,
    temperature: float    = 0.0,
    seed: int             = 1234,
    max_output_tokens: int= 2048
) -> str:
    # 始终使用 SDK 默认 endpoint
    client = genai.Client(api_key=api_key)

    # 把 system-role 的文本和其他消息分离
    system_texts = []
    user_texts   = []
    for msg in chat_history:
        # 安全地拼接每个 part.text，None 视为空串
        parts = msg.get("parts", [])
        text = "".join((part.text or "") for part in parts)
        if msg.get("role") == "system":
            system_texts.append(text)
        else:
            user_texts.append(text)

    config = GenerateContentConfig(
        system_instruction=" ".join(system_texts) or None,
        temperature=temperature,
        seed=seed,
        max_output_tokens=max_output_tokens
    )

    while True:
        try:
            resp = client.models.generate_content(
                model=model,
                contents=user_texts,
                config=config
            )
            return resp.text
        except Exception as e:
            print(f"Gemini API Error: {e}, retrying in 1s…")
            time.sleep(1)


def inference_chat_V2(
    chat_history,
    model: str             = MODEL,
    api_url: str           = None,
    api_key: str           = None,
    temperature: float     = 0.2,
    seed: int              = 42,
    max_output_tokens: int = 2048
) -> str:
    client = genai.Client(api_key=api_key)

    system_texts = []
    user_texts   = []
    for msg in chat_history:
        # 同样处理可能为 None 的 part.text
        parts = msg.get("parts", [])
        text = "".join((part.text or "") for part in parts)
        if msg.get("role") == "system":
            system_texts.append(text)
        else:
            user_texts.append(text)

    config = GenerateContentConfig(
        system_instruction=" ".join(system_texts) or None,
        temperature=temperature,
        seed=seed,
        max_output_tokens=max_output_tokens
    )

    while True:
        try:
            resp = client.models.generate_content(
                model=model,
                contents=user_texts,
                config=config
            )
            return resp.text
        except Exception as e:
            print(f"Gemini API Error: {e}, retrying in 1s…")
            time.sleep(1)
