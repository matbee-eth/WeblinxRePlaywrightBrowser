# Weblinx Replaywright Browser Setup Guide

Welcome to the setup guide for the Weblinx Replaywright Browser. Follow these simple steps to get started with the Huggingface TGI server and initiate your model.

## Step 1: Launch the Huggingface TGI Server
```bash
    text-generation-launcher --model-id McGill-NLP/Llama-3-8b-Web
```

To begin, you'll need to launch the Weblinx powered Playwright browser using the following command:

```bash

    MODEL_SLUG=McGill-NLP/Llama-3-8b-Web API_ENDPOINT=http://localhost:3000/v1 python main.py
```
