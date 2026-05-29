# ChatBot — Telegram Chatbot powered by DialoGPT

A conversational chatbot built with HuggingFace's DialoGPT model and deployed on Telegram via the Bot API. Built as a personal learning project to understand how conversational AI works at a technical level — from tokenization and generation parameters to real-world deployment on a live messaging platform.

---

## How It Works

The bot runs a continuous polling loop that listens for incoming messages from Telegram users. When a message is received, the following steps happen:

1. **Tokenization** — The user's input is converted into token IDs using DialoGPT's tokenizer. Each token represents a subword unit that the model understands.
2. **Chat history concatenation** — The current input is appended to the existing conversation history as a single tensor. This is how the model maintains context — it reads the entire conversation as one long sequence on every turn, rather than storing memory separately.
3. **Response generation** — The model generates a response token by token, controlled by the generation parameters described below.
4. **Decoding** — The output token IDs are decoded back into human-readable text and sent back to the user via Telegram's Bot API.

---

## Model

| Parameter | Value |
|-----------|-------|
| Model | `microsoft/DialoGPT-small` |
| Source | HuggingFace Transformers |

DialoGPT-small was chosen over DialoGPT-medium after testing both. While the medium model produced higher quality responses, the inference time was significantly longer — making the conversation feel unnatural. The small model provides a better balance between response quality and latency for a real-time chat experience.

---

## Generation Parameters

These parameters control how the model generates its responses. Understanding how they interact was one of the core learning outcomes of this project.

### Temperature
Controls the randomness of the output. A lower value makes the model more deterministic and focused, while a higher value increases creativity and variability at the cost of coherence.

| Value | Behavior |
|-------|----------|
| Low (e.g. 0.3) | Focused, repetitive, safe responses |
| Medium (e.g. 0.7) | Balanced between coherence and variety |
| High (e.g. 1.2+) | Creative but risks incoherence |

### top_k
Limits the number of candidate tokens the model considers at each generation step. Only the top k most probable tokens are kept before sampling.

- Lower `top_k` = more conservative, predictable output
- Higher `top_k` = wider candidate pool, more varied responses

### top_p (Nucleus Sampling)
Instead of limiting by a fixed number of candidates, `top_p` limits by cumulative probability mass. The model considers the smallest set of tokens whose combined probability exceeds the threshold `p`.

- `top_p = 0.9` means only tokens that together account for 90% of the probability mass are considered
- Works well combined with `top_k` for balanced output quality

### Key Insight
These parameters must work cohesively. Setting an extremely low temperature with a high `top_k` produces contradictory signals. Finding the right combination requires iterative testing based on the desired output behavior.

---

## What I Learned

- **Tokenization** — How raw text is broken into subword tokens and converted to tensors before being fed into the model
- **Chat history as tensor concatenation** — The model has no persistent memory; context is maintained by feeding the full conversation history as a single input on every turn
- **Generation parameter tuning** — How `temperature`, `top_k`, and `top_p` interact and how to tune them for a target output quality
- **Model size vs inference speed tradeoff** — Larger models have a higher quality ceiling but slower inference, which directly impacts user experience in real-time applications
- **Telegram Bot API integration** — How to connect a locally running model to a live messaging platform using polling and webhook patterns

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| HuggingFace Transformers | DialoGPT model and tokenizer |
| python-telegram-bot | Telegram Bot API integration |
| PyTorch | Tensor operations and model inference |

---

## Repository Structure

```
ChatBot/
├── telegram_bot.py       # Main bot script
├── requirements.txt      # Dependencies
└── README.md             # This file
```

---

## References

- [DialoGPT — Microsoft Research](https://huggingface.co/microsoft/DialoGPT-small)
- [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers)
- [Telegram Bot API Documentation](https://core.telegram.org/bots/api)
- [Anthropic — Building with Claude API](https://anthropic.com)
