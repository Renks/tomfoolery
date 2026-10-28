
import re

import time
from pathlib import Path

import keyboard
import mss
import torch
import open_clip
import pyautogui

from PIL import Image


# ------------ SETTINGS ------------

BASE_DIR = Path(__file__).resolve().parent
DATABASE = BASE_DIR / "knowledge_base"
OUTPUT_DIR = BASE_DIR / "captures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
import os
import pytesseract

TESSERACT_DIR = BASE_DIR / "Tesseract-OCR"

pytesseract.pytesseract.tesseract_cmd = str(
    TESSERACT_DIR / "tesseract.exe"
)

os.environ["TESSDATA_PREFIX"] = str(
    TESSERACT_DIR / "tessdata"
)
MOVE_SECONDS = 2.0
START_DELAY = 5
NUM_DOORS = 10

ENTER_DELAY = 0.5
AFTER_SUBMIT_DELAY = 1.0
TYPE_DELAY = 0.08

# Wait before submitting the answer for door 10.
FINAL_DOOR_DELAY = 100

# Results screen: Play Again button.
# Coordinates are for a 1920x1080 screen.
PLAY_AGAIN_X = 1000
PLAY_AGAIN_Y = 900

AUTOCLICKER_TOGGLE_KEY = "f6"
AUTOCLICKER_START_DELAY = 0.5
ALT_TAB_HOLD_DELAY = 0.3
AFTER_SWITCH_CLICK_DELAY = 1.0

# Wait after clicking Play Again.
REPLAY_DELAY = 3

# Character crop: left, top, right, bottom.
CHARACTER_BOX = (850, 260, 1100, 510)

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

pyautogui.FAILSAFE = True


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


# ------------ SCREEN CAPTURE ------------

def capture_screen():
    with mss.MSS() as screen:
        monitor = screen.monitors[1]
        shot = screen.grab(monitor)

    return Image.frombytes(
        "RGB",
        shot.size,
        shot.rgb
    )


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

    print(
        f"Knowledge base ready: {len(set(names))} characters."
    )

    return features, names


# ------------ RECOGNITION ------------

def recognize_character(
    crop,
    model,
    preprocess,
    device,
    reference_features,
    reference_names
):
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



def play_round(
    round_number,
    model,
    preprocess,
    device,
    reference_features,
    reference_names
):
    print("\n" + "=" * 45)
    print(f"STARTING ROUND {round_number}")
    print("=" * 45)

    for door_number in range(1, NUM_DOORS + 1):
        print(
            f"\nRound {round_number} | "
            f"Door {door_number}/{NUM_DOORS}"
        )

        # Move toward the next door.
        move_forward(MOVE_SECONDS)
        time.sleep(0.3)

        # Activate the answer textbox.
        press_enter()
        time.sleep(ENTER_DELAY)

        clear_textbox()

        # Capture the screen without door synchronization.
        screenshot = capture_screen()

        # Save screenshot.
        screenshot_path = OUTPUT_DIR / (
            f"round_{round_number}_door_{door_number}.png"
        )
        screenshot.save(str(screenshot_path))

        # Crop the character.
        crop = crop_character(
            screenshot,
            door_number,
            round_number
        )

        # Recognize the character.
        name, score = recognize_character(
            crop,
            model,
            preprocess,
            device,
            reference_features,
            reference_names
        )

        print(f"\nPredicted answer: {name}")
        print(f"Similarity: {score:.4f}")

        # Wait 90 seconds on the tenth loop iteration.
        if door_number == NUM_DOORS:
            final_delay = 90

            print(
                "\nFINAL DOOR: Waiting 90 seconds "
                "before submitting..."
            )

            for remaining in range(final_delay, 0, -10):
                print(f"{remaining} seconds remaining...")
                time.sleep(min(10, remaining))

            print("Delay complete. Submitting final answer.")

        # Ensure the textbox is empty before typing.
        clear_textbox()

        keyboard.write(name, delay=TYPE_DELAY)
        time.sleep(0.3)

        print(f"Submitting: {name}")
        press_enter()

        time.sleep(AFTER_SUBMIT_DELAY)

    print(f"\nRound {round_number}: all 10 answers submitted.")

# ------------ PLAY AGAIN ------------



def alt_tab_deliberately():
    keyboard.press("alt")
    time.sleep(0.8)  # Hold Alt before Tab
    keyboard.press("tab")
    time.sleep(0.8)  # Keep Alt+Tab held together
    keyboard.release("tab")
    time.sleep(0.5)  # Keep Alt held a little longer
    keyboard.release("alt")

def click_play_again():
    print("\nWaiting for the results screen...")
    time.sleep(2)

    # Start AutoClicker BEFORE holding Alt.
    print("Starting AutoClicker with F6...")
    keyboard.press_and_release("f6")
    time.sleep(0.1)

    # Switch away from Roblox while AutoClicker is running.
    alt_tab_deliberately()
    time.sleep(0.1)

    

    # Continue clicking for one second after switching back.
    time.sleep(0.1)

    # Stop AutoClicker with F6.
    keyboard.press_and_release("f6")

    print(f"Waiting {REPLAY_DELAY} seconds for the next round...")
    time.sleep(REPLAY_DELAY)

    print("Starting the next round.")



# ------------ MAIN ------------

def main():
    model, preprocess, device = load_model()

    reference_features, reference_names = (
        load_knowledge_base(
            model,
            preprocess,
            device
        )
    )

    print("\nModel and knowledge base are ready.")
    print("Open Roblox and start a round.")

    input(
        "Press Enter here when ready, "
        "then switch to Roblox..."
    )

    print(f"Switch to Roblox within {START_DELAY} seconds.")
    time.sleep(START_DELAY)

    round_number = 1

    # Repeat rounds until manually stopped.
    while True:
        play_round(
            round_number,
            model,
            preprocess,
            device,
            reference_features,
            reference_names
        )

        click_play_again()

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
