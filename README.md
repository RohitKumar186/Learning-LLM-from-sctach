
# Learning LLM from Scratch
This repository tracks my journey of building and understanding Large Language Models (LLMs) from the ground up. Instead of just using APIs, I am implementing the core components—starting from tokenization and data preprocessing to the final transformer architecture.

# Project Overview: Custom Tokenizer (V1)
In the first phase of this project, I developed a Simple Word-Level Tokenizer (SimpleTokenizerV1). This implementation focuses on:

Text Preprocessing: Using Regex to handle punctuation and whitespace with precision.

Vocabulary Mapping: Building a bidirectional mapping between unique tokens and integer IDs.

Special Tokens: Implementing <|unk|> for handling out-of-vocabulary words and <|endoftext|> to mark the boundaries between different text sequences.

Encoding & Decoding: Efficiently converting raw strings into numerical tensors and back into human-readable text.

### Tech Stack
Language: Python 3.x

Libraries: re (Regex for text splitting), Jupyter Notebook (.ipynb)

Dataset: The Verdict by Edith Wharton (for initial vocabulary building)


## Byte Pair Encoding (BPE) & Tiktoken
In this stage of the project, I implemented text tokenization using OpenAI's tiktoken library. Tokenization is the process of converting raw text into a sequence of integers that an LLM can process.

### Why Tiktoken?

Speed: Built in Rust, making it significantly faster than standard Python-based BPE tokenizers.

Efficiency: Uses Byte Pair Encoding to break down common words into single tokens while handling rare words by splitting them into smaller "sub-word" chunks.



# Creating Input-Target Pairs for LLMs
In this stage of the project, I implemented a robust data pipeline to prepare raw text for an autoregressive (GPT-style) model. The core challenge is converting a continuous string of tokens into pairs of Input (what the model sees) and Target (what the model must predict).

1. ## The Sliding Window Approach
LLMs learn by predicting the "next token" in a sequence. To train them efficiently, we use a sliding window to create multiple training examples from a single text.

Input (X): A chunk of tokens of length max_length.

Target (Y): The same chunk, but shifted by one position to the right.

2. ## Key Hyperparameters Implemented
I learned how to control the data flow using these three critical parameters:

Max Length (Context Window): How many tokens the model looks at to make a prediction. (e.g., if max_length=4, the model uses 4 words to guess the 5th).

Stride: How much the window "slides" forward for the next batch.

Low Stride (e.g., 1): High data overlap, but risk of overfitting.

High Stride (e.g., = Max Length): No overlap, ensures the model sees every word once without repetition.

Batch Size: How many input-target pairs the model processes simultaneously before updating its weights.

3.
4. ## PyTorch Data Pipeline
I moved beyond simple Python loops to a professional Dataset/DataLoader architecture:

GPTDatasetV1 (Dataset Class): Handles the logic of chunking the text and converting slices into PyTorch Tensors.

DataLoader: Handles the "heavy lifting" like shuffling the data, organizing it into batches, and using multi-processing (num_workers) to speed up training.

## Code Implementation Check
The implementation successfully generates tensors where:

Input Batch: tensor([[ 290, 4920, 2241,  287]])

Target Batch: tensor([[4920, 2241,  287,  257]])

This confirms that for every token in the input, the corresponding index in the target is exactly the next word in the story!


# Token Embeddings: The Bridge from Numbers to Meaning
This section of the repository covers the transition from discrete Token IDs to continuous Vector Embeddings. While computers process integers, Large Language Models require a higher-dimensional space to understand the semantic relationships between words.

## Key Concepts Covered
### Beyond One-Hot Encoding
Why don't we just use One-Hot Encoding or random IDs?

Problem: Random IDs (e.g., Cat=34, Kitten=92) don't tell the model that a cat and a kitten are related.

Solution: Embeddings exploit the "inherent advantage" of text—semantic meaning. Similar words are placed closer together in a high-dimensional vector space.

### The Embedding Layer as a Lookup Table
In PyTorch, torch.nn.Embedding is technically a "Lookup Table."

Input: A Token ID (index).

Operation: The layer retrieves the corresponding row from a weight matrix.

Architecture: For a gpt2-sized model, this matrix has a Vocabulary Size of 50,257 and an Embedding Dimension of 768.

### Vector Arithmetic (The "Aha!" Moment)
One of the most fascinating proofs of successful embedding training is vector math:

King - Man + Woman ≈ Queen

This demonstrates that the vectors have encoded abstract concepts like "masculinity" and "femininity" into specific dimensions.
