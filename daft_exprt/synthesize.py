import argparse
import json
import logging
import os
import random
import sys
import time
import collections
import re
import uuid

import torch
import librosa
import numpy as np
from scipy.io import wavfile

from shutil import copyfile
from shutil import rmtree
from copy import deepcopy

MAX_WAV_VALUE = 32768.0

FILE_ROOT = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.dirname(FILE_ROOT)
FILE_ROOT = os.path.join(FILE_ROOT, "tmp")
os.makedirs(FILE_ROOT, exist_ok=True)

os.environ['PYTHONPATH'] = PROJECT_ROOT
sys.path.append(PROJECT_ROOT)

from daft_exprt.extract_features import extract_energy, extract_pitch, mel_spectrogram_HiFi, rescale_wav_to_float32
from daft_exprt.hparams import HyperParams
from daft_exprt.model import DaftExprt
from daft_exprt.cleaners import collapse_whitespace, text_cleaner
from daft_exprt.symbols import ascii, eos, punctuation, whitespace
from daft_exprt.utils import chunker

from hifi_gan.models import Generator
from hifi_gan import AttrDict

_logger = logging.getLogger(__name__)
random.seed(1234)

'''
    Script example that showcases how to generate with Daft-Exprt
    using a target sentence, a target speaker, and a target prosody
'''

def get_model(chkpt_path, hparams):
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        checkpoint_dict = torch.load(chkpt_path, map_location=f'cuda:{0}')
    else:
        checkpoint_dict = torch.load(chkpt_path, map_location=f'cpu')

    # hparams = HyperParams(verbose=False, **checkpoint_dict['config_params'])
    # load model
    if gpu_available:
        torch.cuda.set_device(0)
        model = DaftExprt(hparams).cuda(0)
    else:
        model = DaftExprt(hparams)

    state_dict = {k.replace('module.', ''): v for k, v in checkpoint_dict['state_dict'].items()}
    model.load_state_dict(state_dict)    

    # # define cudnn variables
    # random.seed(hparams.seed)
    # torch.manual_seed(hparams.seed)
    # torch.backends.cudnn.deterministic = True
    # _logger.warning('You have chosen to seed training. This will turn on the CUDNN deterministic setting, '
    #                 'which can slow down your training considerably! You may see unexpected behavior when '
    #                 'restarting from checkpoints.\n')
    
    return model


def vocoder_infer(mels, vocoder, lengths=None):

    with torch.no_grad():
        wavs = vocoder(mels).squeeze(1)

    wavs = (
        wavs.cpu().numpy() * MAX_WAV_VALUE
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs


def get_vocoder(config_path, chkpt_path):
    with open(config_path, "r") as f:
        config = json.load(f)
        config = AttrDict(config)

    vocoder = Generator(config)

    if torch.cuda.is_available():
        ckpt = torch.load(chkpt_path)
    else:
        ckpt = torch.load(chkpt_path, map_location=torch.device('cpu'))
    
    vocoder.load_state_dict(ckpt["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()

    if torch.cuda.is_available():
        vocoder.to(f"cuda:{0}")

    return vocoder


def get_dictionary(hparams):
    dictionary = hparams.mfa_dictionary
    # load dictionary and extract word transcriptions
    word_trans = collections.defaultdict(list)
    with open(dictionary, 'r', encoding='utf-8') as f:
        lines = [line.strip().split() for line in f.readlines()] 
    for line in lines:
        word_trans[line[0].lower()].append(line[1:])
    return word_trans

def phonemize_sentence(sentence, dictionary, hparams):
    ''' Phonemize sentence using MFA
    '''
    # get MFA variables
    word_trans = dictionary
    g2p_model = hparams.mfa_g2p_model

    # characters to consider in the sentence
    if hparams.language == 'english':
        all_chars = ascii + punctuation
    else:
        raise NotImplementedError()
    
    # clean sentence
    # "that's, an 'example! ' of a sentence. '"
    sentence = text_cleaner(sentence.strip(), hparams.language).lower().strip()
    # split sentence:
    # [',', "that's", ',', 'an', "example'", '!', "'", 'of', 'a', 'sentence', '.', '.', '.', "'"]
    sent_words = re.findall(f"[\w']+|[{punctuation}]", sentence.lower().strip())
    # remove characters that are not letters or punctuation:
    # [',', "that's", ',', 'an', "example'", '!', 'of', 'a', 'sentence', '.', '.', '.']
    sent_words = [x for x in sent_words if len(re.sub(f'[^{all_chars}]', '', x)) != 0]
    # be sure to begin the sentence with a word and not a punctuation
    # ["that's", ',', 'an', "example'", '!', 'of', 'a', 'sentence', '.', '.', '.']
    while sent_words[0] in punctuation:
        sent_words.pop(0)
    # keep only one punctuation type at the end
    # ["that's", ',', 'an', "example'", '!', 'of', 'a', 'sentence']
    punctuation_end = "."
    while sent_words[-1] in punctuation:
        punctuation_end = sent_words.pop(-1)
    sent_words.append(punctuation_end)
    
    _words = deepcopy(sent_words)

    # phonemize words and add word boundaries
    sentence_phonemized, unk_words = [], []
    while len(sent_words) != 0:
        word = sent_words.pop(0)
        if word in word_trans:
            phones = random.choice(word_trans[word])
            sentence_phonemized.append(phones)
        else:
            unk_words.append(word)
            sentence_phonemized.append('<unk>')
        # at this point we pass to the next word
        # we must add a word boundary between two consecutive words
        print("sent_words: ", sentence_phonemized)
        if len(sent_words) != 0:
            word_bound = sent_words.pop(0) if sent_words[0] in punctuation else whitespace
            sentence_phonemized.append(word_bound)
    # add EOS token
    sentence_phonemized.append(eos)
    
    # use MFA g2p model to phonemize unknown words
    if len(unk_words) != 0:
        rand_name = str(uuid.uuid4())
        oovs = os.path.join(FILE_ROOT, f'{rand_name}_oovs.txt')
        with open(oovs, 'w', encoding='utf-8') as f:
            for word in unk_words:
                f.write(f'{word}\n')
        # generate transcription for unknown words
        oovs_trans = os.path.join(FILE_ROOT, f'{rand_name}_oovs_trans.txt')
        tmp_dir = os.path.join(FILE_ROOT, f'{rand_name}')
        os.system(f'mfa g2p {g2p_model} {oovs} {oovs_trans} -t {tmp_dir}')
        # extract transcriptions
        with open(oovs_trans, 'r', encoding='utf-8') as f:
            lines = [line.strip().split() for line in f.readlines()]
        for line in lines:
            transcription = line[1:]
            unk_idx = sentence_phonemized.index('<unk>')
            sentence_phonemized[unk_idx] = transcription
        # remove files
        os.remove(oovs)
        os.remove(oovs_trans)
        rmtree(tmp_dir, ignore_errors=True)

    print("-- WORDS -- ", _words)
    nb_symbols = 0
    word_idx = 0
    idxs = []
    words = []
    phones = []
    ignore_idxs = []
    for item in sentence_phonemized:
        print("-- item --: ", item)
        if isinstance(item, list):  # correspond to phonemes of a word
            nb_symbols += len(item)
            idxs.append(nb_symbols)
            words.append(_words[word_idx])
            phones.extend(item)
            word_idx += 1
        else:  # correspond to word boundaries
            nb_symbols += 1
            idxs.append(nb_symbols)
            words.append(item)
            if item in _words[word_idx:]:
                word_idx += 1
            phones.append(item)
            ignore_idxs.append(nb_symbols)
        

    print("WORDS!!! ", words)


    return sentence_phonemized, words, phones, idxs, ignore_idxs


def prepare_sentences_for_inference(sentences, dictionary, hparams):
    """
    Phonemize and format sentences to synthesize
    """
    phonemized_sents = []
    sentences = [s.strip() for s in sentences]
    for sent in sentences:
        ps, words, phones, idxs, ignore_idxs  = phonemize_sentence(sent, dictionary, hparams)
        phonemized_sents.append((ps, words, phones, idxs, ignore_idxs))
    return phonemized_sents


def extract_reference_parameters(audio_ref, hparams):
    ''' Extract energy, pitch and mel-spectrogram parameters from audio
        Save numpy arrays to .npz file
    '''
    # read wav file to range [-1, 1] in np.float32
    wav, fs = librosa.load(audio_ref, sr=hparams.sampling_rate)
    wav = rescale_wav_to_float32(wav)
    # get log pitch
    # pitch = extract_pitch(wav, fs, hparams)
    pitch = extract_pitch(wav, hparams)
    # extract mel-spectrogram
    mel_spec = mel_spectrogram_HiFi(wav, hparams)
    # get energy
    energy = extract_energy(np.exp(mel_spec))
    # check sizes are correct
    assert(len(pitch) == mel_spec.shape[1]), f'{len(pitch)} -- {mel_spec.shape[1]}'
    assert(len(energy) == mel_spec.shape[1]), f'{len(energy)} -- {mel_spec.shape[1]}'

    return {"pitch": pitch, "energy": energy, "mel_spec": mel_spec}

# generate mel-specs and synthesize audios with Griffin-Lim
def generate_mel_specs(model, sentences, speaker_ids, refs,
                       hparams, dur_factors, energy_factors, pitch_factors, batch_size, file_names, fine_control=False):
    model.eval()
    # set default values if prosody factors are None
    dur_factors = [None for _ in range(len(sentences))] if dur_factors is None else dur_factors
    energy_factors = [None for _ in range(len(sentences))] if energy_factors is None else energy_factors
    pitch_factors = ['add', [None for _ in range(len(sentences))]] if pitch_factors is None else pitch_factors
    # get pitch transform
    pitch_transform = pitch_factors[0].lower()
    pitch_factors = pitch_factors[1]
    assert(pitch_transform in ['add', 'multiply']), _logger.error(f'Pitch transform "{pitch_transform}" is not currently supported')
    # check lists have the same size
    
    assert (len(speaker_ids) == len(sentences)), _logger.error(f'{len(speaker_ids)} speaker IDs but there are {len(sentences)} sentences to generate')
    assert (len(refs) == len(sentences)), _logger.error(f'{len(refs)} references but there are {len(sentences)} sentences to generate')
    assert (len(dur_factors) == len(sentences)), _logger.error(f'{len(dur_factors)} duration factors but there are {len(sentences)} sentences to generate')
    assert (len(energy_factors) == len(sentences)), _logger.error(f'{len(energy_factors)} energy factors but there are {len(sentences)} sentences to generate')
    assert (len(pitch_factors) == len(sentences)), _logger.error(f'{len(pitch_factors)} pitch factors but there are {len(sentences)} sentences to generate')
    
    predictions = {}
    with torch.no_grad():
        
        for batch_sentences, batch_refs, batch_dur_factors, batch_energy_factors, \
            batch_pitch_factors, batch_speaker_ids, batch_file_names in \
                zip(chunker(sentences, batch_size), chunker(refs, batch_size), 
                    chunker(dur_factors, batch_size), chunker(energy_factors, batch_size),
                    chunker(pitch_factors, batch_size), chunker(speaker_ids, batch_size),
                    chunker(file_names, batch_size)):
        
            batch_predictions =  generate_batch_mel_specs(
                                        model, batch_sentences, batch_refs, batch_dur_factors,
                                        batch_energy_factors, batch_pitch_factors, pitch_transform,
                                        batch_speaker_ids, batch_file_names, hparams, fine_control=fine_control)

            predictions.update(batch_predictions)

    return batch_predictions

def collate_tensors(batch_sentences, batch_dur_factors, batch_energy_factors,
                    batch_pitch_factors, pitch_transform, batch_refs,
                    batch_speaker_ids, batch_file_names, hparams):
    ''' Extract PyTorch tensors for each sentence and collate them for batch generation
    '''
    # gather batch
    batch = []
    for sentence, dur_factors, energy_factors, pitch_factors, refs in \
        zip(batch_sentences, batch_dur_factors, batch_energy_factors, batch_pitch_factors, batch_refs):
            # encode input text as a sequence of int symbols
            symbols = []
            for item in sentence:
                if isinstance(item, list):  # correspond to phonemes of a word
                    symbols += [hparams.symbols.index(phone) for phone in item]
                else:  # correspond to word boundaries
                    symbols.append(hparams.symbols.index(item))
            symbols = torch.IntTensor(symbols)  # (L, )
            # extract duration factors
            if dur_factors is None:
                dur_factors = [1. for _ in range(len(symbols))]
            dur_factors = torch.FloatTensor(dur_factors)  # (L, )
            assert(len(dur_factors) == len(symbols)), \
                _logger.error(f'{len(dur_factors)} duration factors whereas there a {len(symbols)} symbols')
            # extract energy factors
            if energy_factors is None:
                energy_factors = [1. for _ in range(len(symbols))]
            energy_factors = torch.FloatTensor(energy_factors)  # (L, )
            assert(len(energy_factors) == len(symbols)), \
                _logger.error(f'{len(energy_factors)} energy factors whereas there a {len(symbols)} symbols')
            # extract pitch factors
            if pitch_factors is None:
                if pitch_transform == 'add':
                    pitch_factors = [0. for _ in range(len(symbols))]
                elif pitch_transform == 'multiply':
                    pitch_factors = [1. for _ in range(len(symbols))]
            pitch_factors = torch.FloatTensor(pitch_factors)  # (L, )
            assert(len(pitch_factors) == len(symbols)), \
                _logger.error(f'{len(pitch_factors)} pitch factors whereas there a {len(symbols)} symbols')
            # extract references
            # refs = np.load(refs)
            energy_ref, pitch_ref, mel_spec_ref = refs['energy'], refs['pitch'], refs['mel_spec']
            energy_ref = torch.from_numpy(energy_ref).float()  # (T_ref, )
            pitch_ref = torch.from_numpy(pitch_ref).float()  # (T_ref, )
            mel_spec_ref = torch.from_numpy(mel_spec_ref).float()  # (n_mel_channels, T_ref)
            # gather data
            batch.append([symbols, dur_factors, energy_factors, pitch_factors, energy_ref, pitch_ref, mel_spec_ref])
    
    # find symbols sequence max length
    input_lengths, ids_sorted_decreasing = \
        torch.sort(torch.LongTensor([len(x[0]) for x in batch]), dim=0, descending=True)
    max_input_len = input_lengths[0]
    # right pad sequences to max input length
    symbols = torch.LongTensor(len(batch), max_input_len).zero_()
    dur_factors = 1 + torch.FloatTensor(len(batch), max_input_len).zero_()
    energy_factors = 1 + torch.FloatTensor(len(batch), max_input_len).zero_()
    if pitch_transform == 'add':
        pitch_factors = torch.FloatTensor(len(batch), max_input_len).zero_()
    elif pitch_transform == 'multiply':
        pitch_factors = 1 + torch.FloatTensor(len(batch), max_input_len).zero_()
    
    # fill padded arrays
    for i in range(len(ids_sorted_decreasing)):
        # extract batch sequences
        symbols_seq = batch[ids_sorted_decreasing[i]][0]
        dur_factors_seq = batch[ids_sorted_decreasing[i]][1]
        energy_factors_seq = batch[ids_sorted_decreasing[i]][2]
        pitch_factors_seq = batch[ids_sorted_decreasing[i]][3]
        # add sequences to padded arrays
        symbols[i, :symbols_seq.size(0)] = symbols_seq
        dur_factors[i, :dur_factors_seq.size(0)] = dur_factors_seq
        energy_factors[i, :energy_factors_seq.size(0)] = energy_factors_seq
        pitch_factors[i, :pitch_factors_seq.size(0)] = pitch_factors_seq
    
    # find reference max length
    max_ref_len = max([x[6].size(1) for x in batch])
    # right zero-pad references to max output length
    energy_refs = torch.FloatTensor(len(batch), max_ref_len).zero_()
    pitch_refs = torch.FloatTensor(len(batch), max_ref_len).zero_()
    mel_spec_refs = torch.FloatTensor(len(batch), hparams.n_mel_channels, max_ref_len).zero_()
    ref_lengths = torch.LongTensor(len(batch))
    # fill padded arrays
    for i in range(len(ids_sorted_decreasing)):
        # extract batch sequences
        energy_ref_seq = batch[ids_sorted_decreasing[i]][4]
        pitch_ref_seq = batch[ids_sorted_decreasing[i]][5]
        mel_spec_ref_seq = batch[ids_sorted_decreasing[i]][6]
        # add sequences to padded arrays
        energy_refs[i, :energy_ref_seq.size(0)] = energy_ref_seq
        pitch_refs[i, :pitch_ref_seq.size(0)] = pitch_ref_seq
        mel_spec_refs[i, :, :mel_spec_ref_seq.size(1)] = mel_spec_ref_seq
        ref_lengths[i] = mel_spec_ref_seq.size(1)
    
    # reorganize speaker IDs and file names
    file_names = []
    speaker_ids = torch.LongTensor(len(batch))
    for i in range(len(ids_sorted_decreasing)):
        file_names.append(batch_file_names[ids_sorted_decreasing[i]])
        speaker_ids[i] = batch_speaker_ids[ids_sorted_decreasing[i]]
    
    if torch.cuda.is_available():
        _logger.info("GPU available, Moving Tensors to GPU!")
        symbols, dur_factors, energy_factors, pitch_factors, input_lengths, \
        energy_refs, pitch_refs, mel_spec_refs, ref_lengths, speaker_ids = to_gpu(
            symbols, dur_factors, energy_factors, pitch_factors, input_lengths, 
           energy_refs, pitch_refs, mel_spec_refs, ref_lengths, speaker_ids
        )

    return symbols, dur_factors, energy_factors, pitch_factors, input_lengths, \
        energy_refs, pitch_refs, mel_spec_refs, ref_lengths, speaker_ids, file_names


def to_gpu(symbols, dur_factors, energy_factors, pitch_factors, input_lengths, 
           energy_refs, pitch_refs, mel_spec_refs, ref_lengths, speaker_ids):
    # put tensors on GPU
    gpu = next(model.parameters()).device
    symbols = symbols.cuda(gpu, non_blocking=True).long()  # (B, L_max)
    dur_factors = dur_factors.cuda(gpu, non_blocking=True).float()  # (B, L_max)
    energy_factors = energy_factors.cuda(gpu, non_blocking=True).float()  # (B, L_max)
    pitch_factors = pitch_factors.cuda(gpu, non_blocking=True).float()  # (B, L_max)
    input_lengths = input_lengths.cuda(gpu, non_blocking=True).long()  # (B, )
    energy_refs = energy_refs.cuda(gpu, non_blocking=True).float()  # (B, T_max)
    pitch_refs = pitch_refs.cuda(gpu, non_blocking=True).float()  # (B, T_max)
    mel_spec_refs = mel_spec_refs.cuda(gpu, non_blocking=True).float()  # (B, n_mel_channels, T_max)
    ref_lengths = ref_lengths.cuda(gpu, non_blocking=True).long()  # (B, )
    speaker_ids = speaker_ids.cuda(gpu, non_blocking=True).long()  # (B, )

    return symbols, dur_factors, energy_factors, pitch_factors, input_lengths, \
        energy_refs, pitch_refs, mel_spec_refs, ref_lengths, speaker_ids

def to_cpu(duration_preds, durations_int, energy_preds, pitch_preds, 
                        input_lengths, mel_spec_preds, output_lengths, weights):
    # transfer data to cpu and convert to numpy array
    duration_preds = duration_preds.detach().cpu().numpy()  # (B, L_max)
    durations_int = durations_int.detach().cpu().numpy()  # (B, L_max)
    energy_preds = energy_preds.detach().cpu().numpy()  # (B, L_max)
    pitch_preds = pitch_preds.detach().cpu().numpy()  # (B, L_max)
    input_lengths = input_lengths.detach().cpu().numpy()  # (B, )
    mel_spec_preds = mel_spec_preds.detach().cpu().numpy()  # (B, n_mel_channels, T_max)
    output_lengths = output_lengths.detach().cpu().numpy()  # (B)
    weights = weights.detach().cpu().numpy()  # (B, L_max, T_max)

    return duration_preds, durations_int, energy_preds, pitch_preds, input_lengths, \
            mel_spec_preds, output_lengths, weights

def generate_batch_mel_specs(model, batch_sentences, batch_refs, batch_dur_factors,
                                batch_energy_factors, batch_pitch_factors, pitch_transform,
                                batch_speaker_ids, batch_file_names, hparams, fine_control=False):

    # collate batch tensors
    symbols, dur_factors, energy_factors, pitch_factors, input_lengths, \
        energy_refs, pitch_refs, mel_spec_refs, ref_lengths, speaker_ids, file_names = \
            collate_tensors(batch_sentences, batch_dur_factors, batch_energy_factors,
                            batch_pitch_factors, pitch_transform, batch_refs,
                            batch_speaker_ids, batch_file_names, hparams)
    
    # perform inference
    inputs = (symbols, dur_factors, energy_factors, pitch_factors, input_lengths,
              energy_refs, pitch_refs, mel_spec_refs, ref_lengths, speaker_ids)
    

    encoder_preds, decoder_preds, alignments = model.inference(inputs, pitch_transform, hparams, fine_control=fine_control)

    # parse outputs
    duration_preds, durations_int, energy_preds, pitch_preds, input_lengths = encoder_preds
    mel_spec_preds, output_lengths = decoder_preds
    weights = alignments

    # transfer data to cpu and convert to numpy array
    # duration_preds, durations_int, energy_preds, pitch_preds, \
    # input_lengths, mel_spec_preds, output_lengths, weights = to_cpu(duration_preds, 
    #             durations_int, energy_preds, pitch_preds, input_lengths, mel_spec_preds, output_lengths, weights)
    
    # save preds for each element in the batch
    predictions = {}
    for line_idx in range(mel_spec_preds.shape[0]):
        # crop prosody preds to the correct length
        duration_pred = duration_preds[line_idx, :input_lengths[line_idx]]  # (L, )
        duration_int = durations_int[line_idx, :input_lengths[line_idx]]  # (L, )
        energy_pred = energy_preds[line_idx, :input_lengths[line_idx]]  # (L, )
        pitch_pred = pitch_preds[line_idx, :input_lengths[line_idx]]  # (L, )
        # crop mel-spec to the correct length
        mel_spec_pred = mel_spec_preds[line_idx, :, :output_lengths[line_idx]]  # (n_mel_channels, T)
        # crop weights to the correct length
        weight = weights[line_idx, :input_lengths[line_idx], :output_lengths[line_idx]]
        # save generated spectrogram
        file_name = file_names[line_idx]
        # store predictions 
        predictions[f'{file_name}'] = [duration_pred, duration_int, energy_pred, pitch_pred, mel_spec_pred, weight]
    return predictions

def synthesize(model,vocoder, phonemeized_sents, hparams, pitch_factor=None, dur_factor=None, energy_factor=None, fine_control=False):
    # style_bank = os.path.join(PROJECT_ROOT, 'scripts', 'style_bank', 'english')
    ref_path = "/scratch/space1/tc046/lordzuko/work/data/raw_data/BC2013_daft_orig/CB/wavs/CB-EM-04-96.wav"
    ref_parameters = extract_reference_parameters(ref_path, hparams)

    # dur_factor = 1 #1.25  # decrease speed
    pitch_transform = 'add'  # pitch shift
    # pitch_factor = 0  # 50Hz
    # energy_factor = 1
    filenames = ["a.wav"]
    # add duration factors for each symbol in the sentence
    dur_factors = [] if dur_factor is not None else None
    energy_factors = [] if energy_factor is not None else None
    pitch_factors = [pitch_transform, []] if pitch_factor is not None else None
    # for sentence in phonemeized_sents:
        # count number of symbols in the sentence
        # nb_symbols = 0
        # for item in sentence:
        #     if isinstance(item, list):  # correspond to phonemes of a word
        #         nb_symbols += len(item)
        #     else:  # correspond to word boundaries
        #         nb_symbols += 1
        # print(nb_symbols)
        # append to lists
    if dur_factors is not None:
        dur_factors = dur_factor
    if energy_factors is not None:
        energy_factors = energy_factor
    if pitch_factors is not None:
        pitch_factors[1] = pitch_factor

    speaker_ids = [0]*len(phonemeized_sents)
    refs = [ref_parameters]
    batch_size = 1
    # generate mel-specs and synthesize audios with Griffin-Lim
    batch_predictions = generate_mel_specs(model, phonemeized_sents, speaker_ids, refs,
                       hparams, dur_factors, energy_factors, pitch_factors, batch_size, filenames, fine_control=fine_control)

    # duration_pred, duration_int, energy_pred, pitch_pred    
    control_values = {}
    v = batch_predictions["a.wav"]

    control_values["d"] = v[0].unsqueeze(0).detach().cpu().numpy()
    control_values["e"] = v[2].unsqueeze(0).detach().cpu().numpy()
    control_values["p"] = v[3].unsqueeze(0).detach().cpu().numpy()
    mels = v[4].unsqueeze(0)
    wavs = vocoder_infer(mels, vocoder, lengths=None)
    return control_values, wavs[0]

if __name__ == "__main__":
    chkpt_path = "/work/tc046/tc046/lordzuko/work/daft-exprt/trainings/daft_bc2013_v1/checkpoints/DaftExprt_best"
    vocoder_config_path = "/work/tc046/tc046/lordzuko/work/daft-exprt/hifi_gan/config_v1.json"
    vocoder_chkpt_path = "/work/tc046/tc046/lordzuko/work/daft-exprt/trainings/hifigan/checkpoints/g_00100000"
    daft_config_path = "/work/tc046/tc046/lordzuko/work/speech-editor/conf/daft_config.json"
    hparams = HyperParams(**json.load(open(daft_config_path)))
    random.seed(hparams.seed)
    torch.manual_seed(hparams.seed)
    torch.backends.cudnn.deterministic = True
    _logger.warning('You have chosen to seed training. This will turn on the CUDNN deterministic setting, '
                    'which can slow down your training considerably! You may see unexpected behavior when '
                    'restarting from checkpoints.\n')

    model = get_model(chkpt_path, hparams)
    vocoder = get_vocoder(vocoder_config_path, vocoder_chkpt_path)
    dictionary = get_dictionary(hparams)
    # print(vocoder)
    text = "it is possible to wake up a man who sleeps but not who clings to sleep while fully awake."
    sentences = [text]
    phonemeized_sents = prepare_sentences_for_inference(sentences,dictionary, hparams)
    filenames = ["a.wav"]
    print(phonemeized_sents[0][0])
    print(phonemeized_sents[0][1])
    print(phonemeized_sents[0][2])
    print(phonemeized_sents[0][3])
    print(phonemeized_sents[0][4])
    phonemeized_sents = [phonemeized_sents[0][0]]
    style_bank = os.path.join(PROJECT_ROOT, 'scripts', 'style_bank', 'english')
    # ref_path = "/scratch/space1/tc046/lordzuko/work/data/raw_data/BC2013_daft_orig/CB/wavs/CB-EM-01-05.wav"
    ref_path = "/scratch/space1/tc046/lordzuko/work/data/raw_data/BC2013_daft_orig/CB/wavs/CB-EM-04-96.wav"
    # ref_path = "/scratch/space1/tc046/lordzuko/work/data/raw_data/BC2013_daft_orig/CB/wavs/CB-EM-04-100.wav"
    ref_parameters = extract_reference_parameters(ref_path, hparams)

    dur_factor = 1 #1.25  # decrease speed
    pitch_transform = 'add'  # pitch shift
    pitch_factor = 0  # 50Hz
    energy_factor = 1
    # add duration factors for each symbol in the sentence
    dur_factors = [] if dur_factor is not None else None
    energy_factors = [] if energy_factor is not None else None
    pitch_factors = [pitch_transform, []] if pitch_factor is not None else None
    for sentence in phonemeized_sents:
        # count number of symbols in the sentence
        nb_symbols = 0
        for item in sentence:
            if isinstance(item, list):  # correspond to phonemes of a word
                nb_symbols += len(item)
            else:  # correspond to word boundaries
                nb_symbols += 1
        print(nb_symbols)
        # append to lists
        if dur_factors is not None:
            print("dur_factors")
            dur_factors.append([dur_factor for _ in range(nb_symbols)])
        if energy_factors is not None:
            print("energy_factors")
            energy_factors.append([energy_factor for _ in range(nb_symbols)])
        if pitch_factors is not None:
            print("pitch_factors")
            pitch_factors[1].append([pitch_factor for _ in range(nb_symbols)])

    print("factors: ", len(pitch_factors[1][0]), len(energy_factors[0]), len(dur_factors[0]))
    speaker_ids = [0]*len(sentences)
    refs = [ref_parameters]
    batch_size = 1
    # generate mel-specs and synthesize audios with Griffin-Lim
    batch_predictions = generate_mel_specs(model, phonemeized_sents, speaker_ids, refs,
                       hparams, dur_factors, energy_factors, pitch_factors, batch_size, filenames)
    
    
    for k, v in batch_predictions.items():
        print(v[4].shape)
        mels = v[4].unsqueeze(0) #.transpose(1,2)
        print(mels.shape)
        wavs = vocoder_infer(mels, vocoder, lengths=None)
        
        wavfile.write(os.path.join("./", "{}".format(k)), hparams.sampling_rate, wavs[0])