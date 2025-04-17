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


def add_response(role, prompt, chat_history, image_paths=None):
    """
    Append a response with zero or more images (list of filepaths).
    Reads each image locally and inlines it via Part.from_bytes().
    """
    new_history = copy.deepcopy(chat_history)
    parts = [Part.from_text(text=prompt)]
    for img_path in image_paths or []:
        with open(img_path, "rb") as f:
            img_bytes = f.read()
        parts.append(Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))
    new_history.append({
        "role": role,
        "parts": parts
    })
    return new_history

def add_response_old(role, prompt, chat_history, image_paths=None):
    """
    Legacy single-image version, updated to use inline bytes.
    """
    new_history = copy.deepcopy(chat_history)
    parts = [Part.from_text(text=prompt)]
    if image_paths:
        # assume list of one
        img = image_paths[0]
        with open(img, "rb") as f:
            img_bytes = f.read()
        parts.append(Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))
    new_history.append({
        "role": role,
        "parts": parts
    })
    return new_history

def add_response_two_image(role, prompt, chat_history, image_paths):
    """
    Append a response that includes exactly two images.
    """
    new_history = copy.deepcopy(chat_history)
    parts = [Part.from_text(text=prompt)]
    for img_path in image_paths:
        with open(img_path, "rb") as f:
            img_bytes = f.read()
        parts.append(Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))
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
