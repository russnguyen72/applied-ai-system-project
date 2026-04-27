# PawPal+ Enhanced (Module 4)

This AI-assisted task manager helps the user keep track of pet care tasks for each pet that they may have utilizing AI to help add batches of tasks and/or pets that the user would have to manually input otherwise.

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

## Testing Summary

I tested the function of the AI assistant through 2 different approaches. The first approach was unit tests, which can be found in `tests/test_ai_assistant.py`. These unit tests exist to make sure that each function works in an intended use case, as well as in several edge cases. However, unit tests do not catch everything, so the second approach I used was human verification. I personally interacted with and tested the AI assistant in a variety of different ways, especially with how the UI renders since it can be hard for unit tests to catch those results. Even with the testing, there were still many bugs found that were fixed despite passing unit tests, and most likely, many more undiscovered that even my testing could not find.

## Reflection

This project taught me that AI can be an extremely helpful tool when creating a project, with a big emphasis on the tool part. The AI's helpfulness is limited by the human overseeing and controlling it. If the person using the AI does not have the correct detailed vision, either through lack of planning or lack of experience, then there will still be a painful trial and error process to get the product to a satisfactory state. When it comes to reviewing work to ensure correctness, AI loses a lot of its effectiveness, leaving those whose work mainly centered around ensuring the quality of others' work mainly untouched, or even in a worse state if the AI hallucinates severely and/or frequently.

### Suggestsion from the AI

AI has made a lot of helpful decisions, especially when it came to creating the UML diagram within `ai_pawpal_uml_final.png`. Yet, when trying to recommend a local model to use for the application, it incorrectly suggested a slightly more outdated LLM, instead of the newer versions within the same family, showing its limitiation within the training data that it was provided with.

### Limitations

A large limitation of this model can be its performance, especially when ran on systems without a dedicated GPU. This is because local LLMs require VRAM in order to ensure speedy performance, but with many systems, there is no dedicated GPU and therefore a lack of VRAM that make LLMs responding in a timely manner.

### AI Reliability

Something that surprised me was that the AI made almost no inferences when I prompted it to create a task. The AI assistant was implemented to make sure that every required field had relevant information given to it when deciding to create a task or a pet. This resulted in the AI asking when a task is supposed to start more times than not, as the user usually does not explicitly state that a given reminder could notify a user about a task to be done on the same day of its creation. While this ensured reliability by ensuring every piece of required information lived in the prompt that the user gives, it also shows that the user needs to be specific with the system, otherwise it defaults to asking for clarification.

## Demo Link

https://www.loom.com/share/006aa140c09c4941834bfa3457aa7bd6