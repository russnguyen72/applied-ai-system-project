# PawPal+ Enhanced (Module 4)

This AI-assisted task manager helps the user keep track of pet care tasks for each pet that they may have.

## Overview
### Original PawPal+

The original PawPal+ application was a simple task manager. It allowed users to add the pets they are keeping track of and the tasks that are associated with each pet. It was able to handle repeated tasks and notified the users of any time conflicts tasks may have with each other to be dealt with by the user's discretion.

### Architecture Overview

This program has 3 main parts. They are the PawPal system, the LLM backend, and the AI assistant. The LLM Backend interfaces with the local LLM model provided by Ollama, which is currently set to qwen3.5:4b. This LLM Backend then provides an API for the Ollama model to interact with the rest of the system. The AI assistant gives the Ollama model the tools to interact with the PawPal system when called upon, such as creating pets and tasks. The PawPal system holds the user data: the pets and the tasks assigned to each pet.

## Getting Started
### PawPal+ Setup

```bash
python -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### AI Assistant Setup

Install Ollama from https://ollama.com and run

```bash
ollama pull qwen3.5:4b
```

Once you start the application, Ollama should automatically connect with PawPal+ Enhanced.

## Sample Interactions

Prompt: `remind me to go to the vet`

AI Assistant Output: Clarification prompted for which pets, when, and if this task should be repeated.

---

Prompt: `I just bought a new hamster named Reggie. Create a reminder to refill his food and water every 3 days at 15:00`

AI Assistant Output: The user is prompted to add Reggie as a pet and the food and water refill task to their scheduler. The user can accept or deny any individual contribution by the assistant.

## Design Decisions

I built the system with a local AI model in mind as AI assistants are extremely helpful, but API keys are extremely costly and limited. So, the local API runs on system, only needs setup once, and does not need any additional monitoring. Additionally, the more powerful the AI model hosted on the local system, the stronger the performance of the AI assistant. However, if the system that the local LLM is ran on lacks resources such as VRAM or RAM, then the performance of the AI assistant will be significantly degraded, even with a smaller model from Ollama.