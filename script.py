import pytesseract
import os
import random
import re
import threading
import time
from pathlib import Path

import keyboard
# Removed mss as we are shifting to a D3D background capture pipeline
import torch
import open_clip
import pyautogui
from pynput.mouse import Button, Controller
from PIL import Image

# WINDOWS 11 CAPTURE DEPENDENCIES
import win32gui
import win32ui
import win32con
import numpy as np
from PIL import Image
import ctypes  # Added for native Windows DLL access

# ------------ SETTINGS ------------

BASE_DIR = Path(__file__).resolve().parent
DATABASE = BASE_DIR / "knowledge_base"
OUTPUT_DIR = BASE_DIR / "captures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TESSERACT_DIR = BASE_DIR / "Tesseract-OCR"

pytesseract.pytesseract.tesseract_cmd = str(
    TESSERACT_DIR / "tesseract.exe"
)

os.environ["TESSDATA_PREFIX"] = str(
    TESSERACT_DIR / "tessdata"
)
MOVE_SECONDS = 1.65
START_DELAY = 5
NUM_DOORS = 10

ENTER_DELAY = 0.5
AFTER_SUBMIT_DELAY = 1.0
TYPE_DELAY = 0.08

FINAL_DOOR_DELAY = 100
CLICK_DELAY = 0.5

PLAY_AGAIN_X = 793
PLAY_AGAIN_Y = 625
PLAY_AGAIN_SEARCH_BOX = (350, 520, 300+700, 520+200)

DOOR_COUNTER_BOX = (455, 80, 150, 45)
CORRECT_COUNT_BOX = (830, 85, 115, 45)

GUI_ELEMENTS_DIR = BASE_DIR / "guielements"
PLAY_AGAIN_TEMPLATE_PATH = next(
    iter(sorted(GUI_ELEMENTS_DIR.glob("*"))),
    None,
)

AUTOCLICKER_START_DELAY = 0.5
REPLAY_DELAY = 3
CHARACTER_BOX = (524, 161, 824, 461)

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

pyautogui.FAILSAFE = True

# Global handle for the background capture engine
GAME_HWND = None
D3D_ENGINE = None

# ------------ CONTROLS ------------


def move_forward(seconds):
    keyboard.press("w")
    try:
        time.sleep(seconds)
    finally:
        keyboard.release("w")


def press_enter():
    keyboard.press_and_release("enter")


def clear_textbox():
    keyboard.press_and_release("ctrl+a")
    time.sleep(0.15)
    keyboard.press_and_release("backspace")
    time.sleep(0.2)


def ocr_region(region):
    # REFACTORED: Grabs screenshot from background handle context rather than pyautogui
    screenshot = capture_screen()
    if screenshot is None:
        return ""

    # Left, Top, Width, Height -> Crop requires Box coords (Left, Top, Right, Bottom)
    crop_box = (region[0], region[1], region[0] +
                region[2], region[1] + region[3])
    image = screenshot.crop(crop_box).convert("L")
    image = image.resize((region[2] * 4, region[3] * 4))
    text = pytesseract.image_to_string(image, config="--psm 7")
    return text.strip()


def read_game_progress():
    door_text = ocr_region(DOOR_COUNTER_BOX)
    correct_text = ocr_region(CORRECT_COUNT_BOX)

    door_match = re.search(r"(\d+)\s*/\s*(\d+)", door_text, re.IGNORECASE)
    correct_match = re.search(r"(\d+)\s*correct", correct_text, re.IGNORECASE)

    current_door = int(door_match.group(1)) if door_match else None
    total_doors = int(door_match.group(2)) if door_match else None
    correct_count = int(correct_match.group(1)) if correct_match else None

    print(
        f"HUD OCR: door={current_door}/{total_doors}, "
        f"correct={correct_count}, door_text={door_text!r}, "
        f"correct_text={correct_text!r}"
    )

    return current_door, total_doors, correct_count


def load_play_again_template(model, preprocess, device):
    if PLAY_AGAIN_TEMPLATE_PATH is None:
        raise FileNotFoundError(
            f"No GUI template found in {GUI_ELEMENTS_DIR}."
        )

    with Image.open(PLAY_AGAIN_TEMPLATE_PATH) as image:
        template = image.convert("RGB")

    tensor = preprocess(template).unsqueeze(0).to(device)

    with torch.inference_mode():
        template_features = model.encode_image(tensor)
        template_features = template_features / template_features.norm(
            dim=-1,
            keepdim=True
        )

    return template_features


def detect_play_again_popup(model, preprocess, device, template_features, threshold=0.92):
    screenshot = capture_screen()
    if screenshot is None:
        return False

    region = screenshot.crop(PLAY_AGAIN_SEARCH_BOX)
    region.save(OUTPUT_DIR / "play_again_search_region.png")
    tensor = preprocess(region.convert("RGB")).unsqueeze(0).to(device)

    with torch.inference_mode():
        query = model.encode_image(tensor)
        query = query / query.norm(dim=-1, keepdim=True)
        similarity = (query @ template_features.T)[0][0].item()

    print(f"Play Again popup similarity: {similarity:.4f}")
    return similarity >= threshold


# ------------ BACKGROUND SCREEN CAPTURE ------------

def initialize_capture_engine(window_title="Roblox"):
    """Finds and saves the window handle (HWND) for Roblox."""
    global GAME_HWND
    GAME_HWND = win32gui.FindWindow(None, window_title)
    if not GAME_HWND:
        raise Exception(f"Target window '{window_title}' could not be located. Ensure it is open.")
    print(f"Successfully hooked to window handle: {GAME_HWND}")

def capture_screen():
    """Grabs frame buffer data natively via User32.dll to bypass pywin32 module bugs."""
    global GAME_HWND
    if not GAME_HWND:
        return None
        
    try:
        # Get true client dimensions of the background window
        left, top, right, bot = win32gui.GetClientRect(GAME_HWND)
        w = right - left
        h = bot - top

        # Handle zero-size windows gracefully if minimized completely
        if w <= 0 or h <= 0:
            print("Window has no render area. Ensure it is not minimized to tray/taskbar.")
            return None

        # Bind to the window graphics context
        hwndDC = win32gui.GetWindowDC(GAME_HWND)
        mfcDC  = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()

        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
        saveDC.SelectObject(saveBitMap)

        # FIX: Call PrintWindow directly from Windows user32.dll
        # PW_RENDERFULLCONTENT = 2 forces Windows to render hidden/covered DX11/DirectX buffers
        hdc_handle = saveDC.GetSafeHdc()
        ctypes.windll.user32.PrintWindow(GAME_HWND, hdc_handle, 2)

        # Slice frame buffer into a numpy structure
        bmpinfo = saveBitMap.GetInfo()
        bmpstr = saveBitMap.GetBitmapBits(True)
        
        img = np.frombuffer(bmpstr, dtype='uint8')
        img.shape = (bmpinfo['bmHeight'], bmpinfo['bmWidth'], 4)

        # Clean up Win32 GDI objects to prevent memory leaks
        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(GAME_HWND, hwndDC)

        # Drop the Alpha channel, reverse BGRA to RGB, and wrap into PIL Image
        return Image.fromarray(img[..., :3][..., ::-1])
        
    except Exception as e:
        print(f"Native Window Capture Error: {e}")
        return None

def crop_character(screenshot, door_number, round_number):
    left, top, right, bottom = CHARACTER_BOX

    if (
        right > screenshot.width
        or bottom > screenshot.height
    ):
        raise ValueError(
            f"Crop {CHARACTER_BOX} is outside "
            f"screen size {screenshot.size}."
        )

    crop = screenshot.crop(CHARACTER_BOX)

    crop_path = OUTPUT_DIR / (
        f"round_{round_number}_character_{door_number}.png"
    )
    crop.save(str(crop_path))

    return crop


def is_last_door(screenshot):
    width, height = screenshot.size

    counter = screenshot.crop((
        int(width * 0.28),
        int(height * 0.07),
        int(width * 0.72),
        int(height * 0.20)
    ))

    counter = counter.resize(
        (counter.width * 3, counter.height * 3)
    ).convert("L")

    counter_text = pytesseract.image_to_string(
        counter, config="--psm 6"
    )

    print(f"Detected door counter text: {counter_text.strip()}")

    return bool(re.search(r"\b10\s*/\s*10\b", counter_text))

# ------------ MODEL ------------


def load_model():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model, _, preprocess = (
        open_clip.create_model_and_transforms(
            MODEL_NAME,
            pretrained=PRETRAINED
        )
    )

    model = model.to(device).eval()
    print(f"Model ready on: {device}")
    return model, preprocess, device


def load_knowledge_base(model, preprocess, device):
    if not DATABASE.is_dir():
        raise FileNotFoundError(
            f"Knowledge base not found: {DATABASE}"
        )

    extensions = {
        ".png", ".jpg", ".jpeg", ".webp", ".bmp"
    }

    image_paths = sorted(
        path
        for path in DATABASE.rglob("*")
        if path.suffix.lower() in extensions
    )

    if not image_paths:
        raise ValueError(
            "No reference images found in knowledge_base."
        )

    tensors = []
    names = []

    for path in image_paths:
        name = path.parent.name

        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                tensors.append(preprocess(image))
                names.append(name)

        except Exception as error:
            print(f"Skipping {path}: {error}")

    if not tensors:
        raise ValueError("No usable reference images found.")

    batch = torch.stack(tensors).to(device)
    print(f"Encoding {len(tensors)} reference images...")

    with torch.inference_mode():
        features = model.encode_image(batch)
        features = features / features.norm(
            dim=-1,
            keepdim=True
        )

    print(f"Knowledge base ready: {len(set(names))} characters.")
    return features, names


# ------------ RECOGNITION ------------

def recognize_character(crop, model, preprocess, device, reference_features, reference_names):
    tensor = (
        preprocess(crop.convert("RGB"))
        .unsqueeze(0)
        .to(device)
    )

    with torch.inference_mode():
        query = model.encode_image(tensor)
        query = query / query.norm(
            dim=-1,
            keepdim=True
        )

        scores = (query @ reference_features.T)[0]

    best_scores = {}

    for index, score in enumerate(scores.tolist()):
        name = reference_names[index]

        if (
            name not in best_scores
            or score > best_scores[name]
        ):
            best_scores[name] = score

    ranked = sorted(
        best_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    print("\nTop matches:")
    for name, score in ranked[:3]:
        print(f"  {name}: {score:.4f}")

    if not ranked:
        raise ValueError("No character matches found.")

    return ranked[0][0], ranked[0][1]


# ------------ ONE ROUND ------------

def play_round(round_number, model, preprocess, device, reference_features, reference_names, play_again_template_features):
    print("\n" + "=" * 45)
    print(f"STARTING ROUND {round_number}")
    print("=" * 45)

    for door_number in range(1, NUM_DOORS + 1):
        print(f"\nRound {round_number} | Door {door_number}/{NUM_DOORS}")

        current_door, total_doors, _ = read_game_progress()

        if current_door is not None and current_door != door_number:
            print(f"Game state shows door {current_door}/{total_doors} instead of expected door {door_number}. Continuing from current HUD state.")
            if current_door > door_number:
                continue
                
        for attempt in range(1, 6):
            print(f"Door {door_number} attempt {attempt}/5")
            if detect_play_again_popup(model, preprocess, device, play_again_template_features, threshold=0.80):
                print("Play Again popup visible before moving forward; stopping round.")
                return False
                
            move_forward(MOVE_SECONDS)
            time.sleep(0.3)
            press_enter()
            time.sleep(ENTER_DELAY)
            clear_textbox()
            
            screenshot = capture_screen()
            if screenshot is None:
                print("Could not grab frame buffer. Retrying...")
                continue
                
            screenshot_path = OUTPUT_DIR / f"round_{round_number}_door_{door_number}_attempt_{attempt}.png"
            screenshot.save(str(screenshot_path))
            
            crop = crop_character(screenshot, door_number, round_number)
            name, score = recognize_character(crop, model, preprocess, device, reference_features, reference_names)
            print(f"\nPredicted answer: {name} | Similarity: {score:.4f}")
            
            if door_number == NUM_DOORS:
                final_delay = random.uniform(1.0, 5.0)
                print(f"\nFINAL DOOR: Waiting {final_delay:.2f} seconds before submitting...")
                time.sleep(final_delay)
                
            clear_textbox()
            keyboard.write(name, delay=TYPE_DELAY)
            time.sleep(0.3)
            print(f"Submitting: {name}")
            press_enter()
            time.sleep(AFTER_SUBMIT_DELAY)
            
            observed_door, observed_total, observed_correct = read_game_progress()
            if observed_door is None:
                print("Door OCR could not be read; retrying.")
                continue
                
            if observed_door > door_number or (door_number == NUM_DOORS and observed_door >= door_number):
                print(f"Door advanced successfully: observed={observed_door}/{observed_total}, correct={observed_correct}.")
                break
            print(f"Door did not advance. Observed door {observed_door}/{observed_total}. Retrying...")
        else:
            print(f"Door {door_number} failed to advance after 5 attempts. Stopping.")
            return False
            
    return True

# ------------ PLAY AGAIN ------------

class AutoClicker:
    def __init__(self, delay, button):
        self.delay = delay
        self.button = button
        self.running = False
        self.mouse = Controller()

    def start_clicking(self):
        self.running = True

    def stop_clicking(self):
        self.running = False

    def run(self):
        while self.running:
            self.mouse.click(self.button)
            time.sleep(self.delay)

clicker = AutoClicker(CLICK_DELAY, Button.left)
clicker_thread = None

def start_autoclicker():
    global clicker_thread
    if clicker_thread is None or not clicker_thread.is_alive():
        clicker.stop_clicking()
        clicker_thread = threading.Thread(target=clicker.run, daemon=True)
        clicker.start_clicking()
        clicker_thread.start()

def stop_autoclicker():
    clicker.stop_clicking()

def click_play_again(model, preprocess, device):
    print("Play Again popup detected. Moving to button and clicking...")
    # NOTE: Mouse movement actions require the game window to be the active top-level/visible window
    pyautogui.moveTo(PLAY_AGAIN_X, PLAY_AGAIN_Y, duration=0.12)
    time.sleep(0.1)
    start_autoclicker()
    time.sleep(2.5)
    stop_autoclicker()
    print(f"Waiting {REPLAY_DELAY} seconds for the next round...")
    time.sleep(REPLAY_DELAY)
    return True

# ------------ MAIN ------------

def main():
    model, preprocess, device = load_model()
    reference_features, reference_names = load_knowledge_base(model, preprocess, device)
    print("\nModel and knowledge base are ready.")
    input("Press Enter here when ready, then ensure Roblox is running...")
    
    # Hook the D3D Windows 11 capture pipeline directly into the Roblox window process context
    # If your window has a different handle name, alter "Roblox" to match the execution title string
    initialize_capture_engine(window_title="Roblox")
    play_again_template_features = load_play_again_template(model, preprocess, device)
    round_number = 1
    
    while True:
        round_finished = play_round(
            round_number,
            model,
            preprocess,
            device,
            reference_features,
            reference_names,
            play_again_template_features,
        )
        if round_finished:
            click_play_again(model, preprocess, device)
        elif detect_play_again_popup(model, preprocess, device, play_again_template_features, threshold=0.80):
            click_play_again(model, preprocess, device)
        round_number += 1

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBot stopped by user.")
    finally:
        try:
            keyboard.release("w")
        except Exception:
            pass

