import gradio as gr
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, pipeline

# Load the model and tokenizer
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Define language code to name mapping with flag emojis
language_mapping = {
    "🇺🇸 English": "en_XX",
    "🇮🇷 Persian": "fa_IR",
    "🇷🇺 Russian": "ru_RU",
    "🇫🇷 French": "fr_XX",
    "🇩🇪 German": "de_DE",
    "🇪🇸 Spanish": "es_XX",
    "🇹🇷 Turkish": "tr_TR",
    "🇯🇵 Japanese": "ja_XX",
    "🇮🇹 Italian": "it_IT",
    "🇰🇷 Korean": "ko_KR",
}

def summarize_and_translate(article, source_language, target_language):

    # translate source language to English
    tokenizer.src_lang = language_mapping[source_language]
    encoded_ar = tokenizer(article, return_tensors="pt")
    generated_tokens = model.generate(**encoded_ar, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
    translated_article = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    # summarize article
    summarized_article = summarizer(translated_article, max_length=150, min_length=50, do_sample=False)[0]['summary_text']

    # translate English to target language
    tokenizer.src_lang = "en_XX"
    encoded_ar = tokenizer(summarized_article, return_tensors="pt")
    generated_tokens = model.generate(**encoded_ar, forced_bos_token_id=tokenizer.lang_code_to_id[language_mapping[target_language]])
    translated_summarized_article = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    return translated_summarized_article

# Generate description with supported languages and emojis
supported_languages_description = "This tool supports translation between various languages, including:\n"
for language, _ in language_mapping.items():
    supported_languages_description += f"{language} \n"

# Combine the main description with the supported languages description
description_with_emojis = f"📰✨ Welcome to the Article Summarization and Translation tool! Simply enter your article in the text box, select the language it's written in as the source language, and choose the language you want to translate it into as the target language. Then, click 'Submit' to get a concise summary in your desired language. 🌍📝🔍\n\n{supported_languages_description}"

# Read example articles from files
with open("article1.txt", "r", encoding="utf-8") as file:
    example_article1 = file.read()
 
with open("article2.txt", "r", encoding="utf-8") as file:
    example_article2 = file.read()

iface = gr.Interface(fn=summarize_and_translate, 
                     inputs=["text",
                             gr.Dropdown(choices=list(language_mapping.keys()), label="Source Language"),
                             gr.Dropdown(choices=list(language_mapping.keys()), label="Target Language")], 
                     outputs="text", 
                     title="Article Summarization and Translation", 
                     description=description_with_emojis,
                     examples=[
                         [example_article1, "🇮🇷 Persian", "🇮🇹 Italian"],
                         [example_article2, "🇫🇷 French", "🇪🇸 Spanish"]
                     ])
iface.launch()
