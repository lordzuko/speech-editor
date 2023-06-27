import torch
import yaml
import streamlit as st
from pages.ui import ui

from fs2.utils.model import get_model, get_vocoder

config = yaml.load(
    open("conf/config.yaml", "r"), Loader=yaml.FullLoader
)

class Args:
    restore_step = config["inference"]["restore_step"]
    mode = config["inference"]["mode"]
    pitch_control = config["inference"]["pitch_control"]
    energy_control = config["inference"]["energy_control"]
    duration_control = config["inference"]["duration_control"]
    t = train_config = config["train_config_path"]
    m = model_config = config["model_config_path"]
    p = preprocess_config = config["preprocess_config_path"]
    speaker_id = config["inference"]["speaker_id"]

args = Args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
configs = (preprocess_config, model_config, train_config)

# Get model
print("Loading Model...")
model = get_model(args, configs, device, train=False)
print("Model Loaded")
# Load vocoder
print("Loading Vocoder...")
vocoder = get_vocoder(model_config, device)
print("Vocoder Loaded")

st.set_page_config(
    page_title="Speech Editor",
    page_icon="assets/images/icon.png",
    layout="wide",
    initial_sidebar_state="auto",
)

def main():
    ui()


if __name__ == "__main__":
    main()