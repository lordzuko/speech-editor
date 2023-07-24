import os
import json
import yaml
import torch
import random
import logging

from dotenv import load_dotenv
from daft_exprt.hparams import HyperParams
from daft_exprt.synthesize import get_dictionary
from daft_exprt.symbols import whitespace, punctuation, eos

_logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)

MODE = os.environ["MODE"]
DEBUG = os.environ["DEBUG"] == "True"
# DEBUG=True
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
            try:
                STATS["ps"][x][k]["min"] = (STATS["ps"][x][k]["min"] - m) / s
                STATS["ps"][x][k]["max"] = (STATS["ps"][x][k]["max"] - m) / s
                STATS["ps"][x][k]["-2s"] =  (m - (m+2*s)) / s
                STATS["ps"][x][k]["+2s"] =  (m + (m+2*s)) / s
            except ZeroDivisionError as e:
                STATS["ps"][x][k]["min"] = 0.
                STATS["ps"][x][k]["max"] = 0.
                STATS["ps"][x][k]["-2s"] = 0.
                STATS["ps"][x][k]["+2s"] = 0.
        else:
            try:
                STATS["ps"][x][k]["-2s"] =  round((m - (m+2*s)) / s)
                STATS["ps"][x][k]["+2s"] =  round((m + (m+2*s)) / s)
            except ZeroDivisionError as e:
                STATS["ps"][x][k]["-2s"] =  0.
                STATS["ps"][x][k]["+2s"] =  0.

        if  STATS["ps"][x][k]["-2s"] < STATS["ps"][x][k]["min"]:
            STATS["ps"][x][k]["-2s"] = STATS["ps"][x][k]["min"]
        if  STATS["ps"][x][k]["-2s"] > STATS["ps"][x][k]["max"]:
            STATS["ps"][x][k]["-2s"] = STATS["ps"][x][k]["max"]


STATS["gs"] = json.loads(open(config["stats"]["global_stats"]).read())

hparams = HyperParams(**json.load(open(config["daft_config_path"])))

print(STATS["ps"])
dictionary = get_dictionary(hparams)
# define cudnn variables
random.seed(hparams.seed)
torch.manual_seed(hparams.seed)
torch.backends.cudnn.deterministic = True
_logger.warning('You have chosen to seed training. This will turn on the CUDNN deterministic setting, '
                'which can slow down your training considerably! You may see unexpected behavior when '
                'restarting from checkpoints.\n')
ignore_chars = list(whitespace) + list(punctuation) + list(eos)