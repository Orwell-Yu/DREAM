import copy
from google import genai
from google.genai.types import Part

# Initialize Gemini client
# Replace 'YOUR_API_KEY' with your actual Gemini API key
client = genai.Client(api_key="YOUR_API_KEY")
MODEL = "gemini-2.0-flash-001"


def init_action_chat():
    """
    Create a fresh chat history with a system instruction for action mode.
    """
    return [{
        "role": "system",
        "parts": [
            Part(text="You are a helpful AI PC operating assistant. You need to help me operate the PC to complete the user's instruction.")
        ]
    }]


def init_eval_chat():
    """
    Create a fresh chat history with a system instruction for evaluation mode.
    """
    return [{
        "role": "system",
        "parts": [
            Part(text="You are a helpful AI PC operating assistant. You need to help me operate the PC to complete the user's instruction.")
        ]
    }]


def init_reflect_chat():
    """
    Create a fresh chat history with a system instruction for reflection mode.
    """
    return [{
        "role": "system",
        "parts": [
            Part(text="You are a helpful AI PC operating assistant.")
        ]
    }]


def init_memory_chat():
    """
    Create a fresh chat history with a system instruction for memory mode.
    """
    return [{
        "role": "system",
        "parts": [
            Part(text="You are a helpful AI PC operating assistant.")
        ]
    }]


def add_response_old(role, prompt, chat_history, image=None):
    """
    Append a single-turn response. If `image` is provided (filepath), it is sent as one image part.
    """
    new_history = copy.deepcopy(chat_history)
    parts = [Part(text=prompt)]
    if image:
        parts.append(
            Part.from_local_file(
                file_path=image,
                mime_type="image/jpeg"
            )
        )
    new_history.append({
        "role": role,
        "parts": parts
    })
    return new_history


def add_response(role, prompt, chat_history, image=[]):
    """
    Append a response with zero or more images (list of filepaths).
    """
    new_history = copy.deepcopy(chat_history)
    parts = [Part(text=prompt)]
    for img in image:
        parts.append(
            Part.from_local_file(
                file_path=img,
                mime_type="image/jpeg"
            )
        )
    new_history.append({
        "role": role,
        "parts": parts
    })
    return new_history


def add_response_two_image(role, prompt, chat_history, image):
    """
    Append a response that includes exactly two images (list of two filepaths).
    """
    new_history = copy.deepcopy(chat_history)
    parts = [Part(text=prompt)]
    # first image
    parts.append(
        Part.from_local_file(
            file_path=image[0],
            mime_type="image/jpeg"
        )
    )
    # second image
    parts.append(
        Part.from_local_file(
            file_path=image[1],
            mime_type="image/jpeg"
        )
    )
    new_history.append({
        "role": role,
        "parts": parts
    })
    return new_history


def print_status(chat_history):
    """
    Pretty-print the chat history, showing role and placeholder for images.
    """
    print("*" * 100)
    for msg in chat_history:
        print(f"role: {msg['role']}")
        line = ""
        for part in msg["parts"]:
            if hasattr(part, "text") and part.text is not None:
                line += part.text
            elif getattr(part, "mime_type", "").startswith("image/"):
                line += "<image>"
        print(line)
    print("*" * 100)
