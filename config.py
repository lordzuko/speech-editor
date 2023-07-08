import os
import json
import yaml
import torch
from g2p_en import G2p
from fs2.controlled_synthesis import read_lexicon

from dotenv import load_dotenv
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)


DEBUG = bool(os.environ["DEBUG"])

DB = os.environ["DB"]
DB_HOST = os.environ["DB_HOST"]
USERNAME = "" if os.environ["USERNAME"] == "" else os.environ["USERNAME"]
PASSWORD = "" if os.environ["PASSWORD"] == "" else os.environ["PASSWORD"]

config = yaml.load(
    open("conf/config.yaml", "r"), Loader=yaml.FullLoader
)

STATS = dict()
STATS["ps"] = json.loads(open(config["stats"]["phone_stats"]).read())
# normalize min_max
for x in ["p", "e", "d"]:
    for k, v in STATS["ps"][x].items():
        m = v["mean"]
        s = v["std"]
        if x in ["p", "e"]:
            STATS["ps"][x][k]["min"] = (STATS["ps"][x][k]["min"] - m) / s
            STATS["ps"][x][k]["max"] = (STATS["ps"][x][k]["max"] - m) / s
            STATS["ps"][x][k]["-2s"] =  (m - (m+2*s)) / s
            STATS["ps"][x][k]["+2s"] =  (m + (m+2*s)) / s
        else:
            STATS["ps"][x][k]["-2s"] =  round((m - (m+2*s)) / s)
            STATS["ps"][x][k]["+2s"] =  round((m + (m+2*s)) / s)

        if  STATS["ps"][x][k]["-2s"] < STATS["ps"][x][k]["min"]:
            STATS["ps"][x][k]["-2s"] = STATS["ps"][x][k]["min"]
        if  STATS["ps"][x][k]["-2s"] > STATS["ps"][x][k]["max"]:
            STATS["ps"][x][k]["-2s"] = STATS["ps"][x][k]["max"]


STATS["gs"] = json.loads(open(config["stats"]["global_stats"]).read())

print(STATS["ps"])

class Args:
    restore_step = config["inference_config"]["restore_step"]
    mode = config["inference_config"]["mode"]
    pitch_control = config["inference_config"]["pitch_control"]
    energy_control = config["inference_config"]["energy_control"]
    duration_control = config["inference_config"]["duration_control"]
    t = train_config = config["train_config_path"]
    m = model_config = config["model_config_path"]
    p = preprocess_config = config["preprocess_config_path"]
    speaker_id = config["inference_config"]["speaker_id"]

args = Args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
configs = (preprocess_config, model_config, train_config)

g2p = G2p()
lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])
