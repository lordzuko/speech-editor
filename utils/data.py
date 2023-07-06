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
    st.session_state["app"]["i2p"] = {i: p for p, i in enumerate(st.session_state["app"]["p2i"])}
    st.session_state["app"]["w2i"] = {w: i for i, w in enumerate(words)}
    st.session_state["app"]["i2w"] = {i: w for w, i in enumerate(st.session_state["app"]["w2i"])}
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
        
        # 2            p_predictions,
        # 3            e_predictions,
        # 4            log_d_predictions,
        # 5            d_rounded,
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

            print("\n")
            print("FC-PHONE: ") 
            for k, v in st.session_state["app"]["fc"]["phone"].items():
                print(k, v)
            print("\n")

            st.session_state["app"]["unedited"]["phone"] = {
                "p": output[2].detach().cpu().numpy(),
                "e": output[3].detach().cpu().numpy(),
                "d": output[5].detach().cpu().numpy()
            }

            init_stats()

    return wavdata


def process_edited():

    control_values, batchs = preprocess_single(st.session_state["app"]["text"], 
                                                lexicon, 
                                                g2p, 
                                                args, 
                                                preprocess_config, 
                                                st.session_state["app"]["fc"]["phone"])
    
    print("from edited: ", control_values)

    output, wavdata = synthesize(st.session_state["model"], 
                        configs, 
                        st.session_state["vocoder"], 
                        batchs, 
                        control_values)
    
    return wavdata

def init_stats():

    
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
            st.session_state["app"]["fc"]["scaling"]["p"][0][pi] = d_mean/st.session_state["app"]["fc"]["phone"]["p"][0][pi]
            st.session_state["app"]["fc"]["scaling"]["e"][0][pi] = d_mean/st.session_state["app"]["fc"]["phone"]["e"][0][pi]


def cal_avg(word_idx):

    w2p = st.session_state['app']['w2p']
    phones = st.session_state["app"]["p"]
    # print(itemgetter(*w2p[word_idx])(phones))
    phones = itemgetter(*w2p[word_idx])(phones)
    # print(f"word: {word} phones:{phones}")
    # duration, f0, energy is fc is numpy object so no need for itemgetter
    d = st.session_state["app"]["fc"]["phone"]["d"][0][w2p[word_idx]]
    p = st.session_state["app"]["fc"]["phone"]["p"][0][w2p[word_idx]]
    e = st.session_state["app"]["fc"]["phone"]["e"][0][w2p[word_idx]]
    # print(f"word: {word} d:{d} p: {p} e: {e}")
    # print(f"word: {word} d:{np.mean(d)} p: {np.mean(p)} e: {np.mean(e)}")
    
    return round(np.mean(d)), np.mean(p), np.mean(e)
