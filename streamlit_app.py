import streamlit as st
import requests
from bs4 import BeautifulSoup
import json
from transformers import MarianMTModel, MarianTokenizer
import os
import sentencepiece as spm
import re


# 🔁 Tạo từ điển và lưu file JSON (chạy khi khởi động)
def fetch_and_save_slang_dict():
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
            slang_dict[slang_term.lower()] = explanation

    with open("slang_dict.json", "w", encoding="utf-8") as f:
        json.dump(slang_dict, f, ensure_ascii=False, indent=4)
    return slang_dict


# Cắt bỏ chỉ mục và phần tiếng Anh sau
def clean_term(text):
    # Xoá chỉ mục đầu dòng và phần tiếng Anh
    text = re.sub(r'^\.?\s*\d+(\.\d+)*\s*', '', text)        # ví dụ: ". 1.1. " hoặc "1.2. " → ""
    text = re.sub(r'\s+[-–—]\s+.*$', '', text)               # xóa sau dấu gạch: " – tiếng Anh" → ""
    text = re.sub(r'\s*\(.*?\)', '', text)                   # xóa phần trong ngoặc ( )
    return text.strip().lower()

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

def update_slang_json():
    filename = "slang_dict.json"
    
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            old_dict = json.load(f)
    else:
        old_dict = {}
    
    new_dict = fetch_slang_from_learningvietnamese()
    combined = {**old_dict, **new_dict}
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=4)
    return combined

if __name__ == "__main__":
    updated_dict = update_slang_json()
    print(f"✅ Đã cập nhật {len(updated_dict)} từ lóng vào slang_dict.json")


# 🧠 Load model
@st.cache_resource
def load_model():
    model_name = "Helsinki-NLP/opus-mt-en-vi"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

# 📥 Load hoặc tạo từ điển
def load_slang_dict():
    if not os.path.exists("slang_dict.json"):
        return fetch_and_save_slang_dict()
    with open("slang_dict.json", "r", encoding="utf-8") as f:
        return json.load(f)

# 🔠 Dịch nghĩa
def translate_vi_to_en(text, tokenizer, model):
    inputs = tokenizer([text], return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# 🌐 Streamlit UI
st.title("🇻🇳 Vietnamese Slang Translator 🇺🇸")

slang_dict = load_slang_dict()
tokenizer, model = load_model()

slang_input = st.text_input("Nhập từ lóng:")

if slang_input:
    slang_input = slang_input.lower().strip()
    if slang_input in slang_dict:
        vi_meaning = slang_dict[slang_input]
        en_translation = translate_vi_to_en(vi_meaning, tokenizer, model)

        st.markdown(f"### 🌐 English Translation:\n> {vi_meaning}")
        st.markdown(f"### 📝 Nghĩa tiếng Việt:\n> {en_translation}")
    else:
        st.warning("❗ Không tìm thấy từ lóng này trong từ điển.")
