import os
import time
import copy
import torch
import shutil
from PIL import Image, ImageDraw

import PCAgent.api as gpt_api
import PCAgent.api_gemini as gemini_api

import PCAgent.chat as gpt_chat
import PCAgent.chat_gemini as gemini_chat

from PCAgent.text_localization import ocr
from PCAgent.icon_localization import det
from PCAgent.prompt import get_action_prompt, get_eval_prompt, get_reflect_prompt, get_memory_prompt, get_process_prompt


from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope import snapshot_download, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from dashscope import MultiModalConversation
import dashscope
import concurrent

from pynput.mouse import Button, Controller
import argparse
import pyautogui
import pyperclip
from PCAgent.merge_strategy import merge_boxes_and_texts, merge_all_icon_boxes, merge_boxes_and_texts_new
import warnings
warnings.filterwarnings("ignore")

import re
import random
from PIL import ImageFont
from reward.rewardmodel import RewardModel
from transformers import AutoTokenizer as HFTokenizer

pyautogui.FAILSAFE = False

# Open word and create a new document named "hello, world", and type "hello, world" in it.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Program Start...", flush=True)
# 日志文件路径及初始化日志文件
log_file_path = "pcagent_llm_log.txt"
with open(log_file_path, "w", encoding="utf-8") as f:
    f.write("LLM Log Start\n\n")

# 注意：此处的 log_print 函数会根据用户选择决定是否写入日志文件
def log_print(msg):
    print(msg, flush=True)
    # 如果用户选择了 --direct_print 则仅打印到终端，不写入文件
    if not args.direct_print:
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

def contains_chinese(text):
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
    match = chinese_pattern.search(text)
    return match is not None

def cmyk_to_rgb(c, m, y, k):
    r = 255 * (1.0 - c / 255) * (1.0 - k / 255)
    g = 255 * (1.0 - m / 255) * (1.0 - k / 255)
    b = 255 * (1.0 - y / 255) * (1.0 - k / 255)
    return int(r), int(g), int(b)

def draw_coordinates_boxes_on_image(image_path, coordinates, output_image_path, font_path):
    image = Image.open(image_path)
    width, height = image.size
    draw = ImageDraw.Draw(image)
    total_boxes = len(coordinates)
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(total_boxes)]
    for i, coord in enumerate(coordinates):
        c, m, y, k = colors[i]
        color = cmyk_to_rgb(c, m, y, k)
        draw.rectangle(coord, outline=color, width=int(height * 0.0025))
        font = ImageFont.truetype(font_path, int(height * 0.012))
        text_x = coord[0] + int(height * 0.0025)
        text_y = max(0, coord[1] - int(height * 0.013))
        draw.text((text_x, text_y), str(i + 1), fill=color, font=font)
    image = image.convert('RGB')
    image.save(output_image_path)

# 添加新的命令行参数，使用户能够选择只直接打印到终端
parser = argparse.ArgumentParser(description="PC Agent")
parser.add_argument('--instruction', type=str, default='default')
parser.add_argument('--icon_caption', type=int, default=0)  # 0: w/o icon_caption
parser.add_argument('--location_info', type=str, default='center')  # center or bbox or icon_centor; icon_center: only icon center
parser.add_argument('--use_som', type=int, default=1)  # for action
parser.add_argument('--draw_text_box', type=int, default=0, help="whether to draw text boxes in som.")
parser.add_argument('--font_path', type=str, default="/System/Library/Fonts/Times.ttc")
parser.add_argument('--pc_type', type=str, default="windows")  # windows or mac
parser.add_argument('--api_url', type=str, default="https://api.openai.com/v1/chat/completions", help="GPT-4o api url.")
parser.add_argument('--api_token', type=str, default='sk-...', help="Your GPT-4o api token.")
parser.add_argument('--qwen_api', type=str, default='sk-...', help="Input your Qwen-VL api if icon_caption=1.")
parser.add_argument('--add_info', type=str, default='')
parser.add_argument('--disable_reflection', action='store_true')  
parser.add_argument('--disable_memory', action='store_true')
parser.add_argument('--disable_eval', action='store_true')
parser.add_argument('--enable_reward', action='store_true')
# 新增参数：如果选择了 direct_print，则只打印到终端，不写入文件日志
parser.add_argument('--direct_print', action='store_true', help="If set, only print to terminal without writing log file.")
parser.add_argument('--model_backend', type=str, choices=['gpt','gemini'], default='gpt', help="Choose LLM backend: 'gpt' or 'gemini'.")
parser.add_argument('--gemini_api', type=str, default='AI...', help="Your Gemini API key.")

args = parser.parse_args()
gemini_token = args.gemini_api

# Choose backend
if args.model_backend == 'gemini':
    inference_chat = gemini_api.inference_chat
    inference_chat_V2 = gemini_api.inference_chat_V2

    init_action_chat = gemini_chat.init_action_chat
    init_eval_chat = gemini_chat.init_eval_chat
    init_reflect_chat = gemini_chat.init_reflect_chat
    init_memory_chat = gemini_chat.init_memory_chat
    add_response = gemini_chat.add_response
else:
    inference_chat = gpt_api.inference_chat
    inference_chat_V2 = gpt_api.inference_chat_V2

    init_action_chat = gpt_chat.init_action_chat
    init_eval_chat = gpt_chat.init_eval_chat
    init_reflect_chat = gpt_chat.init_reflect_chat
    init_memory_chat = gpt_chat.init_memory_chat
    add_response = gpt_chat.add_response

if args.pc_type == "mac":
    ctrl_key = "command"
    search_key = ["command", "space"]
    ratio = 2
else:
    ctrl_key = "ctrl"
    search_key = ["win", "s"]
    ratio = 1
    args.font_path = r"C:\Windows\Fonts\times.ttf"

if args.model_backend == 'gemini':
    vl_model_version = 'gemini-2.0-flash-001'
    API_url = None
    token = gemini_token
else:
    vl_model_version = 'gpt-4o'
    API_url = args.api_url
    token = args.api_token

def get_screenshot():
    screenshot = pyautogui.screenshot()
    screenshot.save('screenshot/screenshot.png')
    return

def open_app(name):
    log_print('Action: open %s' % name)
    pyautogui.keyDown(search_key[0])
    pyautogui.keyDown(search_key[1])
    pyautogui.keyUp(search_key[1])
    pyautogui.keyUp(search_key[0])
    if contains_chinese(name):
        pyperclip.copy(name)
        pyautogui.keyDown(ctrl_key)
        pyautogui.keyDown('v')
        pyautogui.keyUp('v')
        pyautogui.keyUp(ctrl_key)
    else:
        pyautogui.typewrite(name)
    time.sleep(1)
    pyautogui.press('enter')

def tap(x, y, count=1):
    x, y = x // ratio, y // ratio
    log_print('Action: click (%d, %d) %d times' % (x, y, count))
    mouse = Controller()
    pyautogui.moveTo(x, y)
    mouse.click(Button.left, count=count)
    return

def shortcut(key1, key2):
    if key1 == 'command' and args.pc_type != "mac":
        key1 = 'ctrl'
    log_print('Action: shortcut %s + %s' % (key1, key2))
    pyautogui.keyDown(key1)
    pyautogui.keyDown(key2)
    pyautogui.keyUp(key2)
    pyautogui.keyUp(key1)
    return

def presskey(key):
    log_print('Action: press %s' % key)
    pyautogui.press(key)

def tap_type_enter(x, y, text):
    x, y = x // ratio, y // ratio
    log_print('Action: click (%d, %d), enter %s and press Enter' % (x, y, text))
    pyautogui.click(x=x, y=y)
    if contains_chinese(text):
        pyperclip.copy(text)
        pyautogui.keyDown(ctrl_key)
        pyautogui.keyDown('v')
        pyautogui.keyUp('v')
        pyautogui.keyUp(ctrl_key)
    else:
        pyautogui.typewrite(text)
    time.sleep(1)
    pyautogui.press('enter')
    return

####################################### Edit your Setting #########################################

if args.instruction != 'default':
    instruction = args.instruction
else:
    instruction = "Using Edge, add a adidas hat under $20 to cart in amazon."

caption_call_method = "api"
caption_model = "qwen-vl-max"
qwen_api = args.qwen_api

if args.add_info == '':
    add_info = '''
    When searching in the browser, click on the search bar at the top.
    The input field in WeChat is near the send button.
    When downloading files in the browser, it's preferred to use keyboard shortcuts.
    '''
else:
    add_info = args.add_info

reflection_switch = True if not args.disable_reflection else False
memory_switch = True if not args.disable_reflection else False
eval_switch = True if not args.disable_eval else False
enable_reward = True if args.enable_reward else False


def get_all_files_in_folder(folder_path):
    file_list = []
    for file_name in os.listdir(folder_path):
        file_list.append(file_name)
    return file_list

def draw_coordinates_on_image(image_path, coordinates):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    point_size = 10
    for coord in coordinates:
        draw.ellipse((coord[0] - point_size, coord[1] - point_size, coord[0] + point_size, coord[1] + point_size), fill='red')
    output_image_path = './screenshot/output_image.png'
    image.save(output_image_path)
    return output_image_path

def draw_rectangles_on_image(image_path, coordinates):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    for coord in coordinates:
        draw.rectangle([coord[0], coord[1]], outline="red", width=2)
    output_image_path = './screenshot/output_image.png'
    image.save(output_image_path)
    return output_image_path

def crop(image, box, i):
    image = Image.open(image)
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    if x1 >= x2 - 10 or y1 >= y2 - 10:
        return
    cropped_image = image.crop((x1, y1, x2, y2))
    cropped_image.save(f"./temp/{i}.png")

def generate_local(tokenizer, model, image_file, query):
    query = tokenizer.from_list_format([
        {'image': image_file},
        {'text': query},
    ])
    response, _ = model.chat(tokenizer, query=query, history=None)
    return response

def process_image(image, query):
    dashscope.api_key = qwen_api
    image = "file://" + image
    messages = [{
        'role': 'user',
        'content': [
            {'image': image},
            {'text': query},
        ]
    }]
    response = MultiModalConversation.call(model=caption_model, messages=messages)
    try:
        response = response['output']['choices'][0]['message']['content'][0]["text"]
    except:
        response = "An icon."
    return response

def generate_api(images, query):
    icon_map = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_image, image, query): i for i, image in enumerate(images)}
        for future in concurrent.futures.as_completed(futures):
            i = futures[future]
            response = future.result()
            icon_map[i + 1] = response
    return icon_map

def split_image_into_4(input_image_path, output_image_prefix):
    img = Image.open(input_image_path)
    width, height = img.size
    sub_width = width // 2
    sub_height = height // 2
    quadrants = [
        (0, 0, sub_width, sub_height),
        (sub_width, 0, width, sub_height),
        (0, sub_height, sub_width, height),
        (sub_width, sub_height, width, height)
    ]
    for i, box in enumerate(quadrants):
        sub_img = img.crop(box)
        sub_img.save(f"{output_image_prefix}_part_{i+1}.png")

def ocr_parallel(img, ocr_detection, ocr_recognition, img_x_list, img_y_list, padding, i):
    width, height = Image.open(img).size
    sub_text, sub_coordinates = ocr(img, ocr_detection, ocr_recognition)
    for coordinate in sub_coordinates:
        coordinate[0] = int(max(0, img_x_list[i] + coordinate[0] - padding))
        coordinate[2] = int(min(width * 2, img_x_list[i] + coordinate[2] + padding))
        coordinate[1] = int(max(0, img_y_list[i] + coordinate[1] - padding))
        coordinate[3] = int(min(height * 2, img_y_list[i] + coordinate[3] + padding))
    sub_text_merge, sub_coordinates_merge = merge_boxes_and_texts_new(sub_text, sub_coordinates)
    log_print('parallel end')
    return sub_text_merge, sub_coordinates_merge

def icon_parallel(img, det, img_x_list, img_y_list, padding, i):
    width, height = Image.open(img).size
    sub_coordinates = det(img, "icon", groundingdino_model)
    for coordinate in sub_coordinates:
        coordinate[0] = int(max(0, img_x_list[i] + coordinate[0] - padding))
        coordinate[2] = int(min(width * 2, img_x_list[i] + coordinate[2] + padding))
        coordinate[1] = int(max(0, img_y_list[i] + coordinate[1] - padding))
        coordinate[3] = int(min(height * 2, img_y_list[i] + coordinate[3] + padding))
    sub_coordinates = merge_all_icon_boxes(sub_coordinates)
    return sub_coordinates

def get_perception_infos(screenshot_file, screenshot_som_file, font_path):
    get_screenshot()
    total_width, total_height = Image.open(screenshot_file).size
    split_image_into_4(screenshot_file, './screenshot/screenshot')
    img_list = ['./screenshot/screenshot_part_1.png', './screenshot/screenshot_part_2.png',
                './screenshot/screenshot_part_3.png', './screenshot/screenshot_part_4.png']
    img_x_list = [0, total_width / 2, 0, total_width / 2]
    img_y_list = [0, 0, total_height / 2, total_height / 2]
    coordinates = []
    texts = []
    padding = total_height * 0.0025

    for i, img in enumerate(img_list):
        width, height = Image.open(img).size
        sub_text, sub_coordinates = ocr(img, ocr_detection, ocr_recognition)
        for coordinate in sub_coordinates:
            coordinate[0] = int(max(0, img_x_list[i] + coordinate[0] - padding))
            coordinate[2] = int(min(total_width, img_x_list[i] + coordinate[2] + padding))
            coordinate[1] = int(max(0, img_y_list[i] + coordinate[1] - padding))
            coordinate[3] = int(min(total_height, img_y_list[i] + coordinate[3] + padding))
        sub_text_merge, sub_coordinates_merge = merge_boxes_and_texts_new(sub_text, sub_coordinates)
        coordinates.extend(sub_coordinates_merge)
        texts.extend(sub_text_merge)
    merged_text, merged_text_coordinates = merge_boxes_and_texts(texts, coordinates)

    coordinates = []
    for i, img in enumerate(img_list):
        width, height = Image.open(img).size
        sub_coordinates = det(img, "icon", groundingdino_model)
        for coordinate in sub_coordinates:
            coordinate[0] = int(max(0, img_x_list[i] + coordinate[0] - padding))
            coordinate[2] = int(min(total_width, img_x_list[i] + coordinate[2] + padding))
            coordinate[1] = int(max(0, img_y_list[i] + coordinate[1] - padding))
            coordinate[3] = int(min(total_height, img_y_list[i] + coordinate[3] + padding))
        sub_coordinates = merge_all_icon_boxes(sub_coordinates)
        coordinates.extend(sub_coordinates)
    merged_icon_coordinates = merge_all_icon_boxes(coordinates)

    if args.draw_text_box == 1:
        rec_list = merged_text_coordinates + merged_icon_coordinates
        draw_coordinates_boxes_on_image(screenshot_file, copy.deepcopy(rec_list), screenshot_som_file, font_path)
    else:
        draw_coordinates_boxes_on_image(screenshot_file, copy.deepcopy(merged_icon_coordinates), screenshot_som_file, font_path)

    mark_number = 0
    perception_infos = []
    for i in range(len(merged_text_coordinates)):
        if args.use_som == 1 and args.draw_text_box == 1:
            mark_number += 1
            perception_info = {"text": "mark number: " + str(mark_number) + " text: " + merged_text[i], "coordinates": merged_text_coordinates[i]}
        else:
            perception_info = {"text": "text: " + merged_text[i], "coordinates": merged_text_coordinates[i]}
        perception_infos.append(perception_info)
    for i in range(len(merged_icon_coordinates)):
        if args.use_som == 1:
            mark_number += 1
            perception_info = {"text": "mark number: " + str(mark_number) + " icon", "coordinates": merged_icon_coordinates[i]}
        else:
            perception_info = {"text": "icon", "coordinates": merged_icon_coordinates[i]}
        perception_infos.append(perception_info)
    
    if args.icon_caption == 1:
        image_box = []
        image_id = []
        for i in range(len(perception_infos)):
            if 'icon' in perception_infos[i]['text']:
                image_box.append(perception_infos[i]['coordinates'])
                image_id.append(i)
        for i in range(len(image_box)):
            crop(screenshot_file, image_box[i], image_id[i])
        images = get_all_files_in_folder(temp_file)
        if len(images) > 0:
            images = sorted(images, key=lambda x: int(x.split('/')[-1].split('.')[0]))
            image_id = [int(image.split('/')[-1].split('.')[0]) for image in images]
            icon_map = {}
            prompt = 'This image is an icon from a computer screen. Please briefly describe the shape and color of this icon in one sentence.'
            if caption_call_method == "local":
                for i in range(len(images)):
                    image_path = os.path.join(temp_file, images[i])
                    icon_width, icon_height = Image.open(image_path).size
                    if icon_height > 0.8 * height or icon_width * icon_height > 0.2 * width * height:
                        des = "None"
                    else:
                        des = generate_local(tokenizer, model, image_path, prompt)
                    icon_map[i+1] = des
            else:
                for i in range(len(images)):
                    images[i] = os.path.join(temp_file, images[i])
                icon_map = generate_api(images, prompt)
            for i, j in zip(image_id, range(1, len(image_id)+1)):
                if icon_map.get(j):
                    perception_infos[i]['text'] += ": " + icon_map[j]

    if args.location_info == 'center':
        for i in range(len(perception_infos)):
            perception_infos[i]['coordinates'] = [int((perception_infos[i]['coordinates'][0]+perception_infos[i]['coordinates'][2])/2), 
                                                    int((perception_infos[i]['coordinates'][1]+perception_infos[i]['coordinates'][3])/2)]
    elif args.location_info == 'icon_center':
        for i in range(len(perception_infos)):
            if 'icon' in perception_infos[i]['text']:
                perception_infos[i]['coordinates'] = [
                    int((perception_infos[i]['coordinates'][0] + perception_infos[i]['coordinates'][2]) / 2),
                    int((perception_infos[i]['coordinates'][1] + perception_infos[i]['coordinates'][3]) / 2)]
    return perception_infos, total_width, total_height

### Load caption model ###
torch.manual_seed(1234)
if caption_call_method == "local":
    if caption_model == "qwen-vl-chat":
        model_dir = snapshot_download('qwen/Qwen-VL-Chat', revision='v1.1.0')
        model = AutoModelForCausalLM.from_pretrained(model_dir, device_map=device, trust_remote_code=True).eval()
        model.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True)
    elif caption_model == "qwen-vl-chat-int4":
        qwen_dir = snapshot_download("qwen/Qwen-VL-Chat-Int4", revision='v1.0.0')
        model = AutoModelForCausalLM.from_pretrained(qwen_dir, device_map=device, trust_remote_code=True, use_safetensors=True).eval()
        model.generation_config = GenerationConfig.from_pretrained(qwen_dir, trust_remote_code=True, do_sample=False)
    else:
        log_print("If you choose local caption method, you must choose the caption model from \"Qwen-vl-chat\" and \"Qwen-vl-chat-int4\"")
        exit(0)
    tokenizer = AutoTokenizer.from_pretrained(qwen_dir, trust_remote_code=True)
elif caption_call_method == "api":
    pass
else:
    log_print("You must choose the caption model call function from \"local\" and \"api\"")
    exit(0)

### Load ocr and icon detection model ###
groundingdino_dir = snapshot_download('AI-ModelScope/GroundingDINO', revision='v1.0.0')
groundingdino_model = pipeline('grounding-dino-task', model=groundingdino_dir)
ocr_detection = pipeline(Tasks.ocr_detection, model='damo/cv_resnet18_ocr-detection-line-level_damo')
ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-document_damo')

thought_history = []
summary_history = []
action_history = []
reflection_thought = ""
summary = ""
action = ""
completed_requirements = ""
memory = ""
insight = ""
temp_file = "temp"
screenshot = "screenshot"

if os.path.exists(temp_file):
    shutil.rmtree(temp_file)
os.mkdir(temp_file)
if not os.path.exists(screenshot):
    os.mkdir(screenshot)
error_flag = False

iter = 0
while True:
    iter += 1
    if iter == 1:
        screenshot_file = "./screenshot/screenshot.png"
        screenshot_som_file = "./screenshot/screenshot_som.png"
        perception_infos, width, height = get_perception_infos(screenshot_file, screenshot_som_file, font_path=args.font_path)
        shutil.rmtree(temp_file)
        os.mkdir(temp_file)

    prompt_action = get_action_prompt(instruction, perception_infos, width, height, thought_history, summary_history, action_history, summary, action, reflection_thought, add_info, error_flag, completed_requirements, memory, args.use_som, args.icon_caption, args.location_info)

    # 第一次规划：生成决策 A
    chat_action = init_action_chat()
    if args.use_som == 1:
        chat_action = add_response("user", prompt_action, chat_action, [screenshot_file, screenshot_som_file])
    else:
        chat_action = add_response("user", prompt_action, chat_action, [screenshot_file])
    output_action = inference_chat(chat_action, vl_model_version, API_url, token)
    thought = output_action.split("### Thought ###")[-1].split("### Action ###")[0].replace("\n", " ").replace(":", "").replace("  ", " ").strip()
    summary = output_action.split("### Operation ###")[-1].replace("\n", " ").replace("  ", " ").strip()
    action = output_action.split("### Action ###")[-1].split("### Operation ###")[0].replace("\n", " ").replace("  ", " ").strip()
    chat_action = add_response("assistant", output_action, chat_action)

    status = "#" * 50 + " Decision A " + "#" * 50
    log_print(status)
    log_print("Decision A output:")
    log_print(output_action)
    log_print("LLM Thought: " + thought)
    log_print('#' * len(status))

    if eval_switch:
        print("Planning Scaling Enabled")
        # 第二次规划：生成决策 B（变量名后加2）
        chat_action2 = init_action_chat()
        if args.use_som == 1:
            chat_action2 = add_response("user", prompt_action, chat_action2, [screenshot_file, screenshot_som_file])
        else:
            chat_action2 = add_response("user", prompt_action, chat_action2, [screenshot_file])
        output_action2 = inference_chat_V2(chat_action2, vl_model_version, API_url, token)
        thought2 = output_action2.split("### Thought ###")[-1].split("### Action ###")[0].replace("\n", " ").replace(":", "").replace("  ", " ").strip()
        summary2 = output_action2.split("### Operation ###")[-1].replace("\n", " ").replace("  ", " ").strip()
        action2 = output_action2.split("### Action ###")[-1].split("### Operation ###")[0].replace("\n", " ").replace("  ", " ").strip()
        chat_action2 = add_response("assistant", output_action2, chat_action2)

        status2 = "#" * 50 + " Decision B " + "#" * 50
        log_print(status2)
        log_print("Decision B output:")
        log_print(output_action2)
        log_print("LLM Thought2: " + thought2)
        log_print('#' * len(status2))

        # 构造决策评估的输入字典
        decisionA = {"action": action, "thought": thought, "summary": summary}
        decisionB = {"action": action2, "thought": thought2, "summary": summary2}

        if enable_reward:
            base_model_name = "qwen/Qwen2.5-0.5B"
            reward_model_dir = "../reward_model_out"
            # 1) 准备 tokenizer
            # 1) 先从 Hugging Face Hub 或本地把 base_model 拿回来
            tokenizer = HFTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
            reward_model = RewardModel(base_model_name)
            reward_model.to(device)

            # 2) 读 checkpoint，但 strict=False，意味着“只加载那些能对上尺寸的权重”
            state = torch.load(os.path.join(reward_model_dir, "pytorch_model.bin"), map_location=device)
            reward_model.load_state_dict(state, strict=False)

            # 3) 切 eval / no grad
            reward_model.eval()
            for p in reward_model.parameters():
                p.requires_grad = False

            textA = (
                f"intent: {instruction}\n"
                f"prev_action: {thought}\n"
                f"predict_action: {action}\n"
            )
            textB = (
                f"intent: {instruction}\n"
                f"prev_action: {thought2}\n"
                f"predict_action: {action2}\n"
            )

            # 编码
            inputsA = tokenizer(
                textA,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            inputsB = tokenizer(
                textB,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )

            # 计算分数
            with torch.no_grad():
                rewardA = reward_model(
                    inputsA["input_ids"].to(device),
                    inputsA["attention_mask"].to(device)
                )
                rewardB = reward_model(
                    inputsB["input_ids"].to(device),
                    inputsB["attention_mask"].to(device)
                )

            # 日志输出
            log_print(f"Reward A: {rewardA.item():.4f}")
            log_print(f"Reward B: {rewardB.item():.4f}")

            # 选分数更高的
            if rewardA >= rewardB:
                chosen_action = action
                log_print("Decision A")
            else:
                chosen_action = action2
                log_print("Decision B")
        else:
            # 调用 get_eval_prompt 构造评估 prompt（要求 GPT-4o 仅输出 "A" 或 "B"）
            judge_prompt = get_eval_prompt(instruction, thought_history, summary_history, action_history, decisionA, decisionB)

            # 调用 GPT-4o API 进行判断
            judge_chat = init_eval_chat()
            judge_chat = add_response("user", judge_prompt, judge_chat)
            judge_output = inference_chat(judge_chat, vl_model_version, API_url, token)
            judge_decision = judge_output.strip().upper()  # 预期输出为 "A" 或 "B"
            log_print("#" * 50 + " Judge Decision " + "#" * 50)
            log_print("Judge Output: " + judge_decision)
            log_print("#" * 50)

            # 根据判断结果选择对应的操作
            chosen_action = action if judge_decision == "A" else action2
    
    else:
        chosen_action = action

    # 执行选定的操作
    if "Double Tap" in chosen_action:
        coordinate = chosen_action.split("(")[-1].split(")")[0].split(", ")
        x, y = int(coordinate[0]), int(coordinate[1])
        tap(x, y, 2)
    elif "Triple Tap" in chosen_action:
        coordinate = chosen_action.split("(")[-1].split(")")[0].split(", ")
        x, y = int(coordinate[0]), int(coordinate[1])
        tap(x, y, 3)
    elif "Tap" in chosen_action:
        coordinate = chosen_action.split("(")[-1].split(")")[0].split(", ")
        x, y = int(coordinate[0]), int(coordinate[1])
        tap(x, y, 1)
    elif "Shortcut" in chosen_action:
        keys = chosen_action.split("(")[-1].split(")")[0].split(", ")
        key1, key2 = keys[0].lower(), keys[1].lower()
        shortcut(key1, key2)
    elif "Press" in chosen_action:
        key = chosen_action.split("(")[-1].split(")")[0]
        presskey(key)
    elif "Open App" in chosen_action:
        app = chosen_action.split("(")[-1].split(")")[0]
        open_app(app)
    elif "Type" in chosen_action:
        try:
            # Find the first opening parenthesis and the corresponding closing parenthesis
            start_coord = chosen_action.find("(")
            end_coord = chosen_action.find(")")

            if start_coord != -1 and end_coord != -1 and end_coord > start_coord:
                # Extract coordinates
                coord_part = chosen_action[start_coord + 1:end_coord]
                coordinate = coord_part.split(",")
                if len(coordinate) == 2:
                    x = int(coordinate[0].strip())
                    y = int(coordinate[1].strip())

                    # Extract text following the coordinates
                    # Find the start of the text after the closing parenthesis
                    text_start_index = end_coord + 1
                    # Skip potential separating characters like comma and space
                    while text_start_index < len(chosen_action) and (chosen_action[text_start_index] == ',' or chosen_action[text_start_index].isspace()):
                        text_start_index += 1

                    text = chosen_action[text_start_index:].strip()

                    # Optional: Remove surrounding quotes if they exist (for robustness)
                    if text.startswith('"') and text.endswith('"'):
                        text = text[1:-1]
                    elif text.startswith("'") and text.endswith("'"):
                        text = text[1:-1]
                    elif text.startswith('[') and text.endswith(']'): # Handle potential [text] format
                        text = text[1:-1] # Assumes format like Type (x, y), [some text]


                    if not text:
                        log_print(f"Warning: Extracted empty text for Type action: {chosen_action}")
                        # Decide how to handle this: maybe skip, maybe try a default action?
                        # For now, we'll proceed but log a warning. You might want to raise an error.

                    log_print(f"Parsed for tap_type_enter: x={x}, y={y}, text='{text}'")
                    tap_type_enter(x, y, text) # Call the function with correctly parsed arguments

                else:
                    log_print(f"Error: Could not parse coordinates correctly from {chosen_action}")
                    error_flag = True # Indicate an error occurred
            else:
                log_print(f"Error: Could not find valid coordinates format '(x, y)' in {chosen_action}")
                error_flag = True # Indicate an error occurred

        except Exception as e:
            log_print(f"Error parsing 'Type' action string: '{chosen_action}'")
            log_print(f"Error details: {e}")
            error_flag = True # Indicate an error occurred
    elif "Stop" in chosen_action:
        break


    time.sleep(2)  # 等待动作执行

    if memory_switch:
        prompt_memory = get_memory_prompt(insight)
        chat_action = add_response("user", prompt_memory, chat_action)
        output_memory = inference_chat(chat_action, vl_model_version, API_url, token)
        chat_action = add_response("assistant", output_memory, chat_action)
        status = "#" * 50 + " Memory " + "#" * 50
        log_print(status)
        log_print("Output Memory:")
        log_print(output_memory)
        log_print('#' * len(status))
        output_memory = output_memory.split("### Important content ###")[-1].split("\n\n")[0].strip() + "\n"
        if "None" not in output_memory and output_memory not in memory:
            memory += output_memory

    last_perception_infos = copy.deepcopy(perception_infos)
    last_screenshot_file = "./screenshot/last_screenshot.png"
    if os.path.exists(last_screenshot_file):
        os.remove(last_screenshot_file)
    os.rename(screenshot_file, last_screenshot_file)
    if args.use_som == 1:
        last_screenshot_som_file = "./screenshot/last_screenshot_som.png"
        if os.path.exists(last_screenshot_som_file):
            os.remove(last_screenshot_som_file)
        os.rename(screenshot_som_file, last_screenshot_som_file)

    perception_infos, width, height = get_perception_infos(screenshot_file, screenshot_som_file, font_path=args.font_path)
    shutil.rmtree(temp_file)
    os.mkdir(temp_file)

    if reflection_switch:
        prompt_reflect = get_reflect_prompt(instruction, last_perception_infos, perception_infos, width, height, summary, action, add_info)
        chat_reflect = init_reflect_chat()
        chat_reflect = add_response("user", prompt_reflect, chat_reflect, [last_screenshot_file, screenshot_file])
        output_reflect = inference_chat(chat_reflect, vl_model_version, API_url, token)
        reflection_thought = output_reflect.split("### Thought ###")[-1].split("### Answer ###")[0].replace("\n", " ").strip()
        reflect = output_reflect.split("### Answer ###")[-1].replace("\n", " ").strip()
        chat_reflect = add_response("assistant", output_reflect, chat_reflect)
        status = "#" * 50 + " Reflection " + "#" * 50
        log_print(status)
        # log_print("Output Reflection:")
        log_print(output_reflect)
        log_print('#' * len(status))
    
        if 'A' in reflect: #success
            thought_history.append(thought)
            summary_history.append(summary)
            action_history.append(action)
            prompt_planning = get_process_prompt(instruction, thought_history, summary_history, action_history, completed_requirements, add_info)
            chat_planning = init_memory_chat()
            chat_planning = add_response("user", prompt_planning, chat_planning)
            output_planning = inference_chat(chat_planning, vl_model_version, API_url, token)
            chat_planning = add_response("assistant", output_planning, chat_planning)
            status = "#" * 50 + " Planning " + "#" * 50
            log_print(status)
            log_print("Output Planning:")
            log_print(output_planning)
            log_print('#' * len(status))
            completed_requirements = output_planning.split("### Completed contents ###")[-1].replace("\n", " ").strip()
            error_flag = False
        elif 'B' in reflect: #fail
            error_flag = True
        elif 'C' in reflect: #no change after action
            error_flag = True
    else:
        thought_history.append(thought)
        summary_history.append(summary)
        action_history.append(action)
        prompt_planning = get_process_prompt(instruction, thought_history, summary_history, action_history, completed_requirements, add_info)
        chat_planning = init_memory_chat()
        chat_planning = add_response("user", prompt_planning, chat_planning)
        output_planning = inference_chat(chat_planning, vl_model_version, API_url, token)
        chat_planning = add_response("assistant", output_planning, chat_planning)
        status = "#" * 50 + " Planning " + "#" * 50
        log_print(status)
        # log_print("Output Planning:")
        log_print(output_planning)
        log_print('#' * len(status))
        completed_requirements = output_planning.split("### Completed contents ###")[-1].replace("\n", " ").strip()
    
    os.remove(last_screenshot_file)
    if args.use_som == 1:
        os.remove(last_screenshot_som_file)
    
    # 打印当前累积的 thought_history（每轮都会越来越长）
    log_print("Current Thought History:")
    for idx, t in enumerate(thought_history, 1):
        log_print(f"  {idx}: {t}")
