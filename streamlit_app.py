import streamlit as st
import requests
from bs4 import BeautifulSoup
import json
from transformers import MarianMTModel, MarianTokenizer
import os
import re

# 🔁 Cắt chỉ mục và phần tiếng Anh sau từ header
def clean_term(text):
    text = re.sub(r'^\s*[\.\d]+(\.\d+)*\s*', '', text)
    text = re.sub(r'\s+[-–—]\s+.*$', '', text)
    text = re.sub(r'\s*\(.*?\)', '', text)
    return text.strip().lower()

# 🌐 Lấy từ lóng từ trang learningvietnamese.edu.vn
def fetch_slang_from_learningvietnamese():
    url = "https://learningvietnamese.edu.vn/blog/speak-vietnamese/vietnamese-slang-words/?lang=en"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    slang_dict = {}
    headers = soup.find_all("h3")
    for header in headers:
        term = clean_term(header.text.strip())
        explanation_tag = header.find_next_sibling("p")
        if explanation_tag:
            explanation = explanation_tag.text.strip()
            if term and explanation:
                slang_dict[term] = explanation
    return slang_dict

# 🌐 Từ trang talkpal.ai
def fetch_slang_from_talkpal():
    url = "https://talkpal.ai/vocabulary/top-10-vietnamese-gen-z-slang-terms-you-need-to-know/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    headers = soup.find_all("h2")
    slang_dict = {}
    for header in headers:
        if header.text.strip().startswith(tuple(str(i) for i in range(1, 11))):
            slang_term = header.text.strip().split(". ", 1)[1]
            next_p = header.find_next_sibling("p")
            explanation = next_p.text.strip() if next_p else ""
            slang_dict[clean_term(slang_term)] = explanation
    return slang_dict

# 🧩 Gộp và lưu slang_dict
def update_slang_json():
    filename = "slang_dict.json"
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            old_dict = json.load(f)
    else:
        old_dict = {}

    talkpal = fetch_slang_from_talkpal()
    learnvn = fetch_slang_from_learningvietnamese()

    combined = {**old_dict, **talkpal, **learnvn}
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=4)
    return combined

# 📥 Load từ điển
def load_slang_dict():
    return update_slang_json()

# 🔠 Dịch tiếng Việt → tiếng Anh
@st.cache_resource
def load_model():
    model_name = "Helsinki-NLP/opus-mt-en-vi"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

def translate_vi_to_en(text, tokenizer, model):
    inputs = tokenizer([text], return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# 🌐 Streamlit UI
st.set_page_config(page_title="Vietnamese Slang Translator")
st.title("🇻🇳 Vietnamese Slang Translator 🇺🇸")
st.write("Nhập hoặc chọn từ lóng tiếng Việt để xem nghĩa và bản dịch tiếng Anh.")

slang_dict = load_slang_dict()
tokenizer, model = load_model()

selected_slang = st.selectbox("Chọn hoặc gõ từ lóng:", options=sorted(slang_dict.keys()))

if selected_slang:
    vi_meaning = slang_dict[selected_slang]
    en_translation = translate_vi_to_en(vi_meaning, tokenizer, model)

    st.markdown(f"### 📝 Nghĩa tiếng Việt:\n> {vi_meaning}")
    st.markdown(f"### 🌐 English Translation:\n> {en_translation}")
