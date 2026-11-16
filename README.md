## Bianca: A Personal, Local, Room Assistant
This project contains all the code for **Bianca's operation**. This is currently a personal project.

## Activation
1. Create a new venv (python 3.12) and download the modules by activating the venv and running `pip install -r requirements.txt`

2. Download the openwakeword's models by running (while the venv is active) `python -c "from openwakeword.utils import download_models; download_models()"`

3. You also need to download the kokoro models (`kokoro-v1.0.onnx` and `voices-v1.0.bin`) and put them in `models/kokoro/`

4. To start Bianca, run the `run_bianca.py` python script.