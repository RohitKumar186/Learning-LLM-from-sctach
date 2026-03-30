
# Learning LLM from Scratch
This repository tracks my journey of building and understanding Large Language Models (LLMs) from the ground up. Instead of just using APIs, I am implementing the core components—starting from tokenization and data preprocessing to the final transformer architecture.

# Project Overview: Custom Tokenizer (V1)
In the first phase of this project, I developed a Simple Word-Level Tokenizer (SimpleTokenizerV1). This implementation focuses on:

Text Preprocessing: Using Regex to handle punctuation and whitespace with precision.

Vocabulary Mapping: Building a bidirectional mapping between unique tokens and integer IDs.

Special Tokens: Implementing <|unk|> for handling out-of-vocabulary words and <|endoftext|> to mark the boundaries between different text sequences.

Encoding & Decoding: Efficiently converting raw strings into numerical tensors and back into human-readable text.

# Tech Stack
Language: Python 3.x

Libraries: re (Regex for text splitting), Jupyter Notebook (.ipynb)

Dataset: The Verdict by Edith Wharton (for initial vocabulary building)
