import sys
import os
#replace the path with your hifigan path to import Generator from models.py 
sys.path.append("hifigan")
import argparse
import torch
from espnet2.bin.tts_inference import Text2Speech
from models import Generator
from scipy.io.wavfile import write
from meldataset import MAX_WAV_VALUE
from env import AttrDict
import json
import yaml
import numpy as np
from text_preprocess_for_inference import TTSDurAlignPreprocessor, CharTextPreprocessor, TTSPreprocessor

SAMPLING_RATE = 22050

def load_hifigan_vocoder(language, gender, device):
    # Load HiFi-GAN vocoder configuration file and generator model for the specified language and gender
    vocoder_config = f"vocoder/{gender}/aryan/hifigan/config.json"
    vocoder_generator = f"vocoder/{gender}/aryan/hifigan/generator"
    # Read the contents of the vocoder configuration file
    with open(vocoder_config, 'r') as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    torch.manual_seed(h.seed)
    # Move the generator model to the specified device (CPU or GPU)
    device = torch.device(device)
    generator = Generator(h).to(device)
    state_dict_g = torch.load(vocoder_generator, device)
    generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()

    # Return the loaded and prepared HiFi-GAN generator model
    return generator


def load_fastspeech2_model(language, gender, device):
    
    #updating the config.yaml fiel based on language and gender
    with open(f"punjabi/{gender}/model/config.yaml", "r") as file:      
     config = yaml.safe_load(file)
    
    current_working_directory = os.getcwd()
    feat="model/feats_stats.npz"
    pitch="model/pitch_stats.npz"
    energy="model/energy_stats.npz"
    
    feat_path=os.path.join(current_working_directory,language,gender,feat)
    pitch_path=os.path.join(current_working_directory,language,gender,pitch)
    energy_path=os.path.join(current_working_directory,language,gender,energy)

    
    config["normalize_conf"]["stats_file"]  = feat_path
    config["pitch_normalize_conf"]["stats_file"]  = pitch_path
    config["energy_normalize_conf"]["stats_file"]  = energy_path
        
    with open(f"punjabi/{gender}/model/config.yaml", "w") as file:
        yaml.dump(config, file)
    
    tts_model = f"punjabi/{gender}/model/model.pth"
    tts_config = f"punjabi/{gender}/model/config.yaml"
    
    
    return Text2Speech(train_config=tts_config, model_file=tts_model, device=device)

def text_synthesis(language, gender, sample_text, vocoder, MAX_WAV_VALUE, device):
    # Perform Text-to-Speech synthesis
    with torch.no_grad():
        # Load the FastSpeech2 model for the specified language and gender
        
        model = load_fastspeech2_model(language, gender, device)
       
        # Generate mel-spectrograms from the input text using the FastSpeech2 model
        out = model(sample_text, decode_conf={"alpha": 1})
        print("TTS Done")  
        x = out["feat_gen_denorm"].T.unsqueeze(0) * 2.3262
        x = x.to(device)
        # Use the HiFi-GAN vocoder to convert mel-spectrograms to raw audio waveforms
        y_g_hat = vocoder(x)
        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio = audio.cpu().numpy().astype('int16')
        return audio


def perform_text_synthesis(text_input, language, gender):
    preprocessed_text, _ = preprocessor.preprocess(text_input, language, gender)
    preprocessed_text = " ".join(preprocessed_text)
    audio = text_synthesis(language, gender, preprocessed_text, vocoder, MAX_WAV_VALUE, device)
    return audio


import streamlit as st
import torch
import numpy as np
from scipy.io.wavfile import write
from text_preprocess_for_inference import CharTextPreprocessor

language = "punjabi"
# gender = 'male'
device = "cuda" if torch.cuda.is_available() else "cpu"

preprocessor = CharTextPreprocessor()

# Streamlit app
st.title("Text to Speech Punjabi Language")

text_input = st.text_area("Enter text")

# Radio button for selecting the gender
gender = st.radio("Select Gender", ("male", "female"))
vocoder = load_hifigan_vocoder(language, gender, device)

import streamlit as st
from io import BytesIO

# Assuming the perform_text_synthesis function returns the audio as a numpy array

if st.button("Convert to Speech"):
    audio = perform_text_synthesis(text_input, language, gender.lower())

    # Convert the audio numpy array to bytes
    audio_bytes = BytesIO()
    write(audio_bytes, SAMPLING_RATE, audio)

    # Display the audio in Streamlit
    st.audio(audio_bytes, format="audio/wav")

# Streamlit footer (optional)
st.text("Powered by Sabudh Interns")







