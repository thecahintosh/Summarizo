import streamlit as st
st.set_page_config(page_title="AI TL;DR Summarizer", page_icon="📰")
from newspaper import Article
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
@st.cache_resource
def load_model():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    return tokenizer, model
tokenizer, model = load_model()
st.title("📰 AI TL;DR Summarizer")
st.markdown("Enter a **news article URL** below, and get an AI-generated summary (using T5 model).")
url = st.text_input("🔗 Enter News Article URL")
if st.button("Summarize") and url:
    with st.spinner("⏳ Fetching and summarizing the article..."):
        try:
            article = Article(url)
            article.download()
            article.parse()

            if not article.text.strip():
                st.error("❌ Couldn't extract meaningful content from the URL.")
            else:
                input_text = "summarize: " + article.text.strip()
                input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
                summary_ids = model.generate(
                    input_ids,
                    max_length=150,
                    min_length=40,
                    length_penalty=2.0,
                    num_beams=4,
                    early_stopping=True
                )

                summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                st.success("✅ Summary:")
                st.write(summary)

        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")
st.markdown("---")

