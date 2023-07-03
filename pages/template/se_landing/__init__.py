import datetime
import torch
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils.audio import save_audio
from utils.session import get_state, init_session_state
from fs2.controlled_synthesis import  preprocess_single, synthesize, preprocess_english
from config import lexicon, g2p, args, preprocess_config, configs

# from st_row_buttons import st_row_buttons
# from st_btn_select import st_btn_select
from fs2.utils.model import get_model, get_vocoder
from fs2.text import sequence_to_text
from config import args, configs, device, model_config, preprocess_config, DEBUG

from utils.models import SEData

from streamlit_tags import st_tags

def handle_submit(data , ques,ans, tagger):
    
    output = SEData(wav_file = dict(data.to_mongo())['wav_file'])
    output.essay_question = ques
    output.essay_answer = ans
    output.essay_link = dict(data.to_mongo())['essay_link']
    output.created_at = datetime.datetime.utcnow()
    output.tagger = tagger
    output.tagged_at = datetime.datetime.utcnow()
    output.save()
    #, set__tagger=tagger, set__tagged_at=datetime.datetime.utcnow()
    data.update(set__tagging_status = 'tagged')
    data.save()
    st.session_state["processed_essay_ids"].append(dict(data.to_mongo())['userId'])
    print('Your submitted response')

# def se_flow():
#     st.header("Essay Question Answer Segregation")
#     doc_type = st.sidebar.selectbox("Document Type", ["", "SOP", "Essay"]).lower()
#     level = st.sidebar.selectbox("Essay Level", ["", "Graduate", "Undergraduate"]).lower()
#     source = st.sidebar.selectbox("Essay Source", ["", "EssayForum", "Writing9"]).lower()
#     # user = st.session_state.get("email_id")
#     user = "lordzuko"
#     # from email_id, let's get the assigned user
    
        
#     if st.session_state.get("is_tagging_started"):
#         try:
#             if st.session_state["get_next_question"]:
#                 st.session_state["info"]["data"] = EssayDump.objects(Q(userId__nin = st.session_state["processed_essay_ids"]) & 
#                                                             Q(tagging_status = 'untagged') &
#                                                             Q(level = level) &
#                                                             Q(document_type = doc_type) & 
#                                                             Q(source = source))[0]
                
#                 st.session_state["info"]["data"].update(set__tagging_status = 'in_progress')

#                 st.session_state["info"]["essay_text"] = "\n\n".join([val['text'] for val in dict(st.session_state["info"]["data"].to_mongo())['data']])
                    
#             st.session_state["get_next_question"] = False
#             st.markdown("**Document**")
#             st.write(st.session_state["info"]["essay_text"])
#             st.markdown("---")

#             # When there are Multiple Prompts and Multiple essays put MQ/MA
#             # When there are multiple essays and one prompt, choose the best essay
#             # When there are multiple prompts, and one essay put MQ for Prompt.
#             # When there is no essay and just prompt paste the prompt, and put NA for the essay and vice versa.
#             # When there is No prompt and No essay put NA in both boxes.
            
#             st.markdown(f"""**Essay ID**: \t\t{dict(st.session_state["info"]["data"].to_mongo())["userId"]}""")
#             ques_ph = st.empty()
#             ans_ph = st.empty()

#             st.session_state["info"]["question"] = ques_ph.text_area("Question", value="", height=30)
#             st.session_state["info"]["answer"] = ans_ph.text_area("Answer", value="", height=100)
            
#             submit_bt = st.button("Submit")
#             st.markdown("---")
#             next_bt = st.button("Next Question")

#             if submit_bt:
#                 with st.spinner("Submitting..."):
#                     handle_submit(st.session_state["info"]["data"] , 
#                                 st.session_state["info"]["question"], 
#                                 st.session_state["info"]["answer"],
#                                 st.session_state["login"]["user_email"])

#                 ques_ph.text_area(label="Question",value="")
#                 ans_ph.text_area(label="Answer", value="")
#                 st.success("Successful")

#             if next_bt:
#                 st.session_state["get_next_question"] = True
#                 st.session_state["info"] = {}
#                 st.experimental_rerun()
#         except IndexError:
#             st.success("No Documents available for tagging")
#     else:
#         st.subheader("Tagging Notes")
#         st.markdown("""
#             * First select the type of essay, source and level for which the tagging has to be performed from the left panel.
#             * Once the correct options have been selected, click, `Start Tagging` to proceed.
#         """)
#         start_bt = st.button("Click here to start tagging")
#         if start_bt:
#             st.session_state["is_tagging_started"] = True
#             st.session_state["get_next_question"] = True
#             st.session_state["info"] = dict()
#             st.session_state["processed_essay_ids"] = []
#             st.experimental_rerun()

def setup_data(texts, words, idxs):

    phones = sequence_to_text(list(texts[0]))
    st.session_state["app"]["p"] = phones
    st.session_state["app"]["w"] = words

    st.session_state["app"]["p2i"] = {p: i for i, p in enumerate(phones)}
    st.session_state["app"]["i2p"] = {i: p for p, i in enumerate(st.session_state["app"]["p2i"])}
    st.session_state["app"]["w2i"] = {w: i for i, w in enumerate(words)}
    st.session_state["app"]["i2w"] = {i: w for w, i in enumerate(st.session_state["app"]["w2i"])}
    # st.session_state["app"]["w2p"] = 

def tag_ui(suggestions, values):
    words_to_edit = st_tags(
        label="Add words to edit",
        text="enter word for suggestion and press enter",
        value=values,
        suggestions=suggestions)
    return words_to_edit

def setup_sliders(column):
        
    cur_word_idx = st.session_state["app"]["current_word_idx"]

    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        duration_control = st.slider("Speed Scale", 
                                            value=float(st.session_state["app"]["fine_control"]["duration"][0][cur_word_idx]), 
                                            min_value=0.0, 
                                            max_value=3.0, 
                                            help="Speech speed. Larger value become slow")
        
    with col2:
        f0_control = st.slider("Pitch Scale", 
                                    value=float(st.session_state["app"]["fine_control"]["f0"][0][cur_word_idx]), 
                                    min_value=0.0, 
                                    max_value=3.0)
        
    
    with col3:
        energy_control = st.slider("Energy Scale", 
                                        value=float(st.session_state["app"]["fine_control"]["energy"][0][cur_word_idx]), 
                                        min_value=0.0, 
                                        max_value=3.0)

    with st.form("Control Variance"):
        
        word = st.session_state["app"]["current_word"]
        
        st.markdown(f"Currently Editing: `{word}` at index `{cur_word_idx}`")
        
        
        
        

        submitted = st.form_submit_button("Synthesize Selected")       
        if submitted:
            st.session_state["app"]["fine_control"]["duration"][0][cur_word_idx] = duration_control
            st.session_state["app"]["fine_control"]["f0"][0][cur_word_idx] = f0_control
            st.session_state["app"]["fine_control"]["energy"][0][cur_word_idx] = energy_control
            
            with column:
                # if DEBUG:
                #     st.json(st.session_state["app"])
                setup_speech_edited()
                with st.expander("Waveform visualization"):
                    fig = plt.figure()
                    ax1 = fig.add_subplot(1, 1, 1)
                    ax1.specgram(st.session_state["app"]["edited"]["wav"],
                                    Fs=st.session_state["sampling_rate"])
                    st.pyplot(fig)



        
def setup_speech_unedited():
    
    if "unedited" not in st.session_state["app"]:
        st.session_state["app"]["unedited"] = {}
        st.session_state["app"]["unedited"]["synthesized"] = False
    
    # if "f0" not in st.session_state["app"]["unedited"]:
    if not st.session_state["app"]["unedited"]["synthesized"]:
        st.session_state["app"]["unedited"]["synthesized"] =  st.button("Synth!", 
                                                        disabled=st.session_state["app"]["unedited"]["synthesized"])
        
        if st.session_state["app"]["unedited"]["synthesized"]:
            print(",asdf", st.session_state["app"]["unedited"])
            with torch.no_grad():

                control_values, batchs = preprocess_single(st.session_state["app"]["text"], 
                                                        lexicon, 
                                                        g2p, 
                                                        args, 
                                                        preprocess_config)
                wavdata = synthesize(st.session_state["model"], 
                                    configs, 
                                    st.session_state["vocoder"], 
                                    batchs, 
                                    control_values)
                            
                
                st.session_state["app"]["unedited"]["wav"] = wavdata
                st.markdown("Original:")
                st.audio(st.session_state["app"]["unedited"]["wav"],
                            sample_rate=st.session_state["sampling_rate"])
    else:
        print("there:", st.session_state["app"]["unedited"])
        st.markdown("Original:")
        st.audio(st.session_state["app"]["unedited"]["wav"],
                            sample_rate=st.session_state["sampling_rate"])
        
        with st.expander("Waveform visualization"):
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 1, 1)
    #         ax1.plot(wavdata)

    #         ax2 = fig.add_subplot(2, 1, 2)
            ax1.specgram(st.session_state["app"]["unedited"]["wav"],
                            Fs=st.session_state["sampling_rate"])
            st.pyplot(fig)



def setup_speech_edited():
    if "edited" not in st.session_state["app"]:
        st.session_state["app"]["edited"] = {}
        st.session_state["app"]["unedited"]["synthesized"] = False

    control_values, batchs = preprocess_single(st.session_state["app"]["text"], 
                                                lexicon, 
                                                g2p, 
                                                args, 
                                                preprocess_config, 
                                                st.session_state["app"]["fine_control"])
    
    wavdata = synthesize(st.session_state["model"], 
                        configs, 
                        st.session_state["vocoder"], 
                        batchs, 
                        control_values)

    st.markdown("Edited:")
    st.session_state["app"]["edited"]["wav"] = wavdata
    st.audio(st.session_state["app"]["edited"]["wav"],
                sample_rate=st.session_state["sampling_rate"])

def save():

    save_audio(st.session_state["app"]["unedited"]["wav"], 
               st.session_state["sampling_rate"], 
               f"data/unedited/{st.session_state['app']['filename']}.wav")
    save_audio(st.session_state["app"]["edited"]["wav"], 
               st.session_state["sampling_rate"], 
               f"data/edited/{st.session_state['app']['filename']}.wav")
    
def reset():
    st.session_state["app"] = {} # this will be used to track the speech data
    st.session_state["app"]["edit_next"] = True
    st.session_state["app"]["begin_processing"] = False
    st.experimental_rerun()

def se_edit_widget():

    if st.button("Made a mistake reset?"):
        reset()

    words_to_edit = []
    
    if st.session_state["app"]["edit_next"]:
        with st.form("Enter details:"):
            st.session_state["app"]["text"] = st.text_area("Text", value="", height=100, max_chars=2048)
            st.session_state["app"]["filename"] = st.text_input("Filename")
            submitted = st.form_submit_button("Submit")
                
                
    if st.session_state["app"].get("text"):
        st.session_state["app"]["begin_processing"] = True
        print("beging: ", st.session_state["app"]["begin_processing"]) 
        print("making edit_text = False")
        st.session_state["app"]["edit_next"] = False

    print("beging:-- ", st.session_state["app"]["begin_processing"]) 

    if st.session_state["app"]["begin_processing"]:
        text = st.session_state["app"]["text"]

        print("beginng!")
        if not st.session_state["app"]["edit_next"]:
            out = preprocess_english(text,lexicon, g2p, preprocess_config)
            texts, words, idxs = np.array([out[0]]), out[1], out[2]

            # st.markdown(texts)
            setup_data(texts, words, idxs)
            # st.markdown(words)
            # st.markdown(idxs)
            st.markdown(f"Text: {st.session_state['app']['text']}")
            st.markdown(f"Filename: {st.session_state['app']['filename']}")

            if "fine_control" not in st.session_state["app"]:
                st.session_state["app"]["fine_control"] = {
                    "f0": np.ones((1,len(words))),
                    "energy": np.ones((1,len(words))),
                    "duration": np.ones((1,len(words)))
                }
            # setup_data
            setup_data(texts, words, idxs)
            
            if not st.session_state["app"].get("suggestions"):
                suggestions = [f"{w}-{i}" for i,w in enumerate(st.session_state["app"]["w"])]
                st.session_state["app"]["suggestions"] = suggestions

            if "words_to_edit" not in st.session_state["app"]:
                st.markdown("init tag ui")
                st.session_state["app"]["words_to_edit"] = tag_ui(st.session_state["app"]["suggestions"], values=st.session_state["app"]["suggestions"])
                
            else:
                st.session_state["app"]["words_to_edit"] = tag_ui(st.session_state["app"]["suggestions"], values=st.session_state["app"]["words_to_edit"])
            # if "words_to_edit" in st.session_state["app"]:
            #     st.markdown("post tag ui")
            #     st.session_state["app"]["words_to_edit"] = tag_ui(st.session_state["app"]["suggestions"], 
            #                            st.session_state["app"]["words_to_edit"])
            
            col1, col2 = st.columns([2, 2])
            with col1:
                setup_speech_unedited()

            if st.session_state["app"]["words_to_edit"]:
                current_word = st.session_state["app"]["words_to_edit"][0]

                word, idx = current_word.split("-")
                

                
                # st.session_state["app"]["next_word"] = False
                # with col1:
                if "unedited" in st.session_state["app"]:
                    if st.session_state["app"]["unedited"]["synthesized"]:
                        st.session_state["app"]["current_word"] = current_word
                        st.session_state["app"]["current_word_idx"] = int(idx)
                        setup_sliders(column=col2)

            if not st.session_state["app"]["words_to_edit"] and "edited" in st.session_state["app"]:

                with col2:
                     st.markdown("Edited:")
                     st.audio(st.session_state["app"]["edited"]["wav"],
                        sample_rate=st.session_state["sampling_rate"])

                with st.form("Finalize editing?:"):
                    st.markdown("Finalize editing?")
                    submitted = st.form_submit_button("Complete")
                    if submitted:
                        save()
                        reset()
                       

                
                
    return words_to_edit

def se_ui():        

    if st.session_state.get("is_tagging_started"):
            print(dict(st.session_state).keys())

            if (not get_state(st, "model")) or (not get_state(st, "vocoder")) or (not get_state(st, "sampling_rate")):
                with st.spinner("Loading and setting up TTS model..."):
                    if not get_state(st, "model"):
                        # Get model
                        print("Loading Model...")
                        init_session_state(st, "model", get_model(args, configs, device, train=False))
                        print("Model Loaded")
                    
                    if not get_state(st, "vocoder"):
                        # Load vocoder
                        print("Loading Vocoder...")
                        init_session_state(st, "vocoder", get_vocoder(model_config, device))
                        print("Vocoder Loaded")

                    if not get_state(st, "sampling_rate"):
                        init_session_state(st, "sampling_rate", preprocess_config["preprocessing"]["audio"]["sampling_rate"])
        
            st.header("Speech Editor")
            se_edit_widget()
    else:
        st.subheader("Tagging Notes")
        st.markdown("""
            * First select the type of essay, source and level for which the tagging has to be performed from the left panel.
            * Once the correct options have been selected, click, `Start Tagging` to proceed.
        """)
        start_bt = st.button("Click here to start tagging")
        if start_bt:
            st.session_state["is_tagging_started"] = True
            reset()



    # autoplay_onoff = st.checkbox("Auto play")
    
    # model_not_loaded = get_state(st, "model") is None

    # ready_for_synthesie = not (model_not_loaded and text != "")

    # print("Here", model_not_loaded, text)

    # if st.button("Synth!", disabled=ready_for_synthesie):
    #     with torch.no_grad():
    #         control_values, batchs = preprocess_single(text, lexicon, g2p, args, preprocess_config)
    #         wavdata = synthesize(get_state(st, "model"), configs, get_state(st, "vocoder"), batchs, control_values)

    #         # ts = get_state(st, "model")(text)
    #         # wavdata = ts.view(-1).cpu().numpy()
    #         init_session_state(st, "wavdata", wavdata)

    # if "wavdata" in st.session_state:
    #     wavdata = get_state(st, 'wavdata')
    #     samplerate = get_state(st, 'sampling_rate')
    #     st.audio(wavdata, sample_rate=samplerate)

    #     if autoplay_onoff is True:
    #         audio_base64 = get_audio_bytes(wavdata, samplerate)
    #         audio_tag = f'<audio autoplay="true" src="data:audio/wav;base64,{audio_base64}">'
    #         st.markdown(audio_tag, unsafe_allow_html=True)

    #     with st.expander("Waveform visualization"):
    #         fig = plt.figure()
    #         ax1 = fig.add_subplot(2, 1, 1)
    #         ax1.plot(wavdata)

    #         ax2 = fig.add_subplot(2, 1, 2)
    #         ax2.specgram(wavdata, Fs=samplerate)
        
    #         st.pyplot(fig)
