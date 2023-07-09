import torch
import numpy as np
from pprint import pprint
import json
import streamlit as st
from operator import itemgetter
from fs2.text import sequence_to_text
from fs2.controlled_synthesis import  preprocess_single, synthesize, preprocess_english
from config import lexicon, g2p, args, preprocess_config, configs, STATS
from config import args, configs, device, model_config, preprocess_config, DEBUG
from copy import deepcopy

def setup_data(texts, words, idxs):
    """
    Setup the application data to deal with words, phone, mapping etc.
    """
    phones = sequence_to_text(list(texts[0]))
    phones = phones.lstrip("{")
    phones = phones.rstrip("}")
    phones = phones.split(" ")
    
    st.session_state["app"]["p"] = phones
    st.session_state["app"]["w"] = words
    st.session_state["app"]["idxs"] = idxs

    st.session_state["app"]["p2i"] = {p: i for i, p in enumerate(phones)}
    st.session_state["app"]["i2p"] = {i: p for i, p in enumerate(phones)}
    st.session_state["app"]["w2i"] = {w: i for i, w in enumerate(words)}
    st.session_state["app"]["i2w"] = {i: w for i, w in enumerate(words)}
    st.session_state["app"]["w2p"] = {}
    c = 0
    for i, w in enumerate(words):
        st.session_state["app"]["w2p"][i] = list(range(c,c+idxs[i]))
        c+=idxs[i]


def variance_control():
    pass

def standardize(x, mean, std):
    return (x - mean) / std

def sample_gauss(x, mean, std):
    return x * std + mean


def process_unedited():
    
    with torch.no_grad():

        control_values, batchs = preprocess_single(st.session_state["app"]["text"], 
                                                lexicon, 
                                                g2p, 
                                                args, 
                                                preprocess_config)
        
        output, wavdata = synthesize(st.session_state["model"], 
                            configs, 
                            st.session_state["vocoder"], 
                            batchs, 
                            control_values)

        if "fc" not in st.session_state["app"]:
            st.session_state["app"]["fc"] = {}
            st.session_state["app"]["fc"]["phone"] = {
                "p": output[2].detach().cpu().numpy(),
                "e": output[3].detach().cpu().numpy(),
                "d": output[5].detach().cpu().numpy()
            }

            if DEBUG:
                print("\n")
                print("Unedited-FC-PHONE: ") 
                for k, v in st.session_state["app"]["fc"]["phone"].items():
                    print(k, v)
                print("\n")

            st.session_state["app"]["unedited"]["phone"] = deepcopy(st.session_state["app"]["fc"]["phone"])
            
            init_stats()

    return wavdata

def prepare_mask():
    d_mask = st.session_state["app"]["unedited"]["phone"]["d"] != st.session_state["app"]["fc"]["phone"]["d"]
    p_mask = st.session_state["app"]["unedited"]["phone"]["p"] != st.session_state["app"]["fc"]["phone"]["p"]
    e_mask = st.session_state["app"]["unedited"]["phone"]["e"] != st.session_state["app"]["fc"]["phone"]["e"]
    
    st.session_state["app"]["fc"]["mask"] = {}
    st.session_state["app"]["fc"]["mask"]["d"] = d_mask
    st.session_state["app"]["fc"]["mask"]["p"] = p_mask
    st.session_state["app"]["fc"]["mask"]["e"] = e_mask

    # return d_scaled_mask, p_scaled_mask, e_scaled_mask



def update_phone_variance():

    for i, _ in enumerate(st.session_state["app"]["w"]):
        d_mean = st.session_state["app"]["fc"]["word"]["d"][0][i]
        p_mean = st.session_state["app"]["fc"]["word"]["p"][0][i]
        e_mean = st.session_state["app"]["fc"]["word"]["e"][0][i]

        for pi in st.session_state["app"]["w2p"][i]:
            val = round(d_mean/st.session_state["app"]["fc"]["scaling"]["d"][0][pi])
            print(pi, st.session_state["app"]["i2p"])
            phone = st.session_state["app"]["i2p"][pi]
            st.session_state["app"]["fc"]["phone"]["d"][0][pi] = val # np.clip(val, a_min=STATS["ps"]["d"][phone]["-2s"], a_max=STATS["ps"]["d"][phone]["+2s"])
            val = p_mean/st.session_state["app"]["fc"]["scaling"]["p"][0][pi]
            st.session_state["app"]["fc"]["phone"]["p"][0][pi] = np.clip(val, a_min=STATS["ps"]["p"][phone]["-2s"], a_max=STATS["ps"]["p"][phone]["+2s"])
            val = e_mean/st.session_state["app"]["fc"]["scaling"]["e"][0][pi]
            st.session_state["app"]["fc"]["phone"]["e"][0][pi] = np.clip(val, a_min=STATS["ps"]["e"][phone]["-2s"], a_max=STATS["ps"]["e"][phone]["+2s"])


def process_edited():

    # update the phone variances based on the changes in word variances
    update_phone_variance()
    if DEBUG:
        # apply scaling
        prepare_mask()

    control_values, batchs = preprocess_single(st.session_state["app"]["text"], 
                                                lexicon, 
                                                g2p, 
                                                args, 
                                                preprocess_config, 
                                                st.session_state["app"]["fc"]["phone"])
    if DEBUG:    
        print("\n")
        print("Edited-FC-PHONE: ") 
        for k, v in st.session_state["app"]["fc"]["phone"].items():
            print(k, v)
        print("\n")
        print("-----------------------------")
    output, wavdata = synthesize(st.session_state["model"], 
                        configs, 
                        st.session_state["vocoder"], 
                        batchs, 
                        control_values)
    
    return wavdata

def init_stats():

    st.session_state["app"]["fc"]["gl_d"] = 0.
    st.session_state["app"]["fc"]["gl_p"] = 0.
    st.session_state["app"]["fc"]["gl_e"] = 0.

    st.session_state["app"]["fc"]["scaling"] = {
        "p": np.ones((1,len(st.session_state["app"]["p"]))),
        "e": np.ones((1,len(st.session_state["app"]["p"]))),
        "d": np.ones((1,len(st.session_state["app"]["p"])))
    }
    st.session_state["app"]["fc"]["word"] = {
        "p": np.ones((1,len(st.session_state["app"]["w"]))),
        "e": np.ones((1,len(st.session_state["app"]["w"]))),
        "d": np.ones((1,len(st.session_state["app"]["w"])))
    }
    st.session_state["app"]["unedited"]["word"] = {
        "p": np.ones((1,len(st.session_state["app"]["w"]))),
        "e": np.ones((1,len(st.session_state["app"]["w"]))),
        "d": np.ones((1,len(st.session_state["app"]["w"])))
    }


    for i, _ in enumerate(st.session_state["app"]["w"]):
        d_mean, p_mean, e_mean = cal_avg(i)

        st.session_state["app"]["unedited"]["word"]["d"][0][i] = d_mean
        st.session_state["app"]["unedited"]["word"]["p"][0][i] = p_mean
        st.session_state["app"]["unedited"]["word"]["e"][0][i] = e_mean

        st.session_state["app"]["fc"]["word"]["d"][0][i] = d_mean
        st.session_state["app"]["fc"]["word"]["p"][0][i] = p_mean
        st.session_state["app"]["fc"]["word"]["e"][0][i] = e_mean

        for pi in st.session_state["app"]["w2p"][i]:
            # scaling factor is avg/curr_phone_val
            # later we can scale curr_phone_val = avg/scaling_factor
            st.session_state["app"]["fc"]["scaling"]["d"][0][pi] = d_mean/st.session_state["app"]["fc"]["phone"]["d"][0][pi]
            st.session_state["app"]["fc"]["scaling"]["p"][0][pi] = p_mean/st.session_state["app"]["fc"]["phone"]["p"][0][pi]
            st.session_state["app"]["fc"]["scaling"]["e"][0][pi] = e_mean/st.session_state["app"]["fc"]["phone"]["e"][0][pi]


def cal_avg(word_idx):

    w2p = st.session_state['app']['w2p']
    # phones = st.session_state["app"]["p"]
    # print(itemgetter(*w2p[word_idx])(phones))
    # phones = itemgetter(*w2p[word_idx])(phones)
    d = st.session_state["app"]["fc"]["phone"]["d"][0][w2p[word_idx]]
    p = st.session_state["app"]["fc"]["phone"]["p"][0][w2p[word_idx]]
    e = st.session_state["app"]["fc"]["phone"]["e"][0][w2p[word_idx]]

    return round(np.mean(d)), np.mean(p), np.mean(e)