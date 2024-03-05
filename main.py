from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, pipeline

file_path = 'article.txt'
with open(file_path, 'r') as file:
    # Read the contents of the file
    article = file.read()

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# translate source language to English
tokenizer.src_lang = "fa_IR"
encoded_ar = tokenizer(article, return_tensors="pt")
generated_tokens = model.generate(**encoded_ar, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
traslated_article = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# summarize article
summarized_article = summarizer(traslated_article, max_length=150, min_length=50, do_sample=False)[0]['summary_text']

# translate English to target language
tokenizer.src_lang = "en_XX"
encoded_ar = tokenizer(summarized_article, return_tensors="pt")
generated_tokens = model.generate(**encoded_ar, forced_bos_token_id=tokenizer.lang_code_to_id["fa_IR"])
traslated_summarized_article = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

file_path = 'summarized_article.txt'
with open(file_path, 'w') as file:
    # Write the contents
    article = file.write(traslated_summarized_article)

