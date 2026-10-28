# tomfoolery
dont even know if i should describe ts tbh
# Roblox Anime Character Guessing Bot

A Python-based automation bot that recognizes anime characters in a Roblox guessing game using computer vision and a pretrained CLIP model. It captures the game screen, identifies the character by comparing it with a local reference-image database, and automatically submits the predicted answer.

## Features

- **AI-powered character recognition** using CLIP ViT-B-32.
- **Image similarity matching** against a local character database.
- **Automated gameplay** using keyboard input simulation.
- **Screenshot capture and cropping** for character identification.
- **OCR support** using Tesseract to read on-screen text.
- **Automated answer submission** with configurable timing.
- **Round automation** for games containing multiple doors.
- **Optional replay automation** using an external AutoClicker configured with F6.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Deep learning framework |
| OpenCLIP | Loads the pretrained CLIP model |
| CLIP ViT-B-32 | Visual feature extraction and image similarity |
| Pillow | Image processing and cropping |
| MSS | Screen capture |
| PyAutoGUI | Mouse and screen automation, if used |
| Keyboard | Keyboard input automation |
| Pytesseract | OCR integration |
| Tesseract OCR | Text recognition engine |

## Model Information

- **Architecture:** CLIP ViT-B-32
- **Pretrained checkpoint:** `laion2b_s34b_b79k`
- **Framework:** PyTorch
- **Implementation:** OpenCLIP
- **Recognition method:** Image embedding similarity

The project uses pretrained weights rather than training a neural network from scratch. The model extracts visual representations from the character screenshot and compares them with representations of reference images in the local database.

The reference image with the highest similarity score is selected as the predicted character.

## Project Structure

```text
roblox-anime-guessing-bot/
├── script.py
├── knowledge_base/
│   ├── character_1/
│   ├── character_2/
│   └── ...
├── captures/
├── Tesseract-OCR/
│   ├── tesseract.exe
│   └── tessdata/
└── README.md
```

The directory structure above is illustrative. Adjust it to match the actual organization of your repository.

- `script.py` — Main automation and recognition script.
- `knowledge_base/` — Reference images used to identify characters.
- `captures/` — Screenshots and character crops generated during execution.
- `Tesseract-OCR/` — Local Tesseract installation, if bundled with the project.

## Requirements

- Windows 10 or Windows 11
- Python 3.10 or another Python version compatible with the installed dependencies
- Roblox installed and running
- An internet connection for initial model-weight downloads, unless the weights are already cached
- Tesseract OCR, if OCR functionality is enabled

An NVIDIA GPU is optional. PyTorch can run on a CPU, although performance may differ.

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd roblox-anime-guessing-bot
```

Replace `<YOUR_REPOSITORY_URL>` with your repository's clone URL.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install torch open_clip_torch Pillow mss PyAutoGUI keyboard pytesseract
```

For GPU acceleration, install a CUDA-enabled PyTorch build compatible with your NVIDIA driver by following the official [PyTorch installation guide](https://pytorch.org/get-started/locally/).

### 4. Configure Tesseract OCR

If the repository includes a local Tesseract installation, make sure the executable and language-data directory are present in the expected locations.

Otherwise, install Tesseract separately and configure its executable path in the Python script.

### 5. Prepare the character database

Place the supported reference images inside `knowledge_base/`, following the naming and folder conventions expected by the recognition code.

Ensure that the script can associate each reference image with the correct character name.

## Running the Bot

1. Launch Roblox and enter the supported guessing game.
2. Ensure the game window and display resolution match the configured screen coordinates.
3. Check the character crop coordinates and timing settings in `script.py`.
4. Open a terminal in the project directory.
5. Run:

```bash
python script.py
```

Keep the game window in the expected state while the bot is running. The automation depends on screen capture, keyboard focus, and configured timing values.

## How It Works

1. **Screen capture:** Captures the current game screen.
2. **Character cropping:** Extracts the region containing the anime character.
3. **Feature extraction:** Uses CLIP to encode the screenshot crop and reference images.
4. **Similarity comparison:** Compares visual embeddings to find the closest reference.
5. **Prediction:** Selects the character associated with the highest similarity score.
6. **Answer submission:** Types the predicted name into the game and submits it.
7. **Round progression:** Repeats the process using configured movement and delay settings.

The bot relies on visual similarity, so predictions can be affected by image quality, character pose, lighting, cropping, and the coverage of the reference database.

## Configuration

Depending on the game and display resolution, you may need to adjust:

- Character crop coordinates
- Movement duration
- Keyboard input delay
- Time between submissions
- Number of doors per round
- Character reference images
- Screenshot output location

The current implementation uses sequential door processing. It does not reliably synchronize every step with the game's displayed door counter.

## Limitations

- Recognition accuracy depends on the quality and variety of reference images.
- Incorrect or ambiguous visual matches can produce incorrect answers.
- Screen coordinates and input timing may need adjustment for different display configurations.
- CPU inference can be slower than GPU inference.
- OCR can fail when text is small, obscured, or surrounded by other interface elements.
- The project is designed for the configured game and may not work unchanged in other Roblox experiences.

## Safety and Responsible Use

This project is an independent experiment in Python automation and computer vision. It is not affiliated with or endorsed by Roblox or OpenAI.

Use it only where permitted by the game's rules and Roblox's applicable policies. Automated gameplay and third-party tools may be restricted.

Do not include account passwords, session cookies, API keys, or other credentials in the source code or repository. Review the repository contents before publishing, and avoid uploading unnecessary personal data or proprietary assets.

## Future Improvements

Potential improvements include:

- Better screenshot preprocessing and character cropping
- More diverse reference images
- Improved OCR and game-state synchronization
- Configurable game resolutions
- Prediction confidence thresholds
- A graphical interface for configuration
- More robust error handling and logging

## Acknowledgements

- [OpenAI CLIP](https://github.com/openai/CLIP) — Original CLIP research and implementation
- [OpenCLIP](https://github.com/mlfoundations/open_clip) — Open-source CLIP implementation
- [PyTorch](https://pytorch.org/) — Deep learning framework
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) — Optical character recognition engine

---

**Disclaimer:** This project is provided for educational and research purposes. Use it responsibly and in accordance with applicable platform rules.
