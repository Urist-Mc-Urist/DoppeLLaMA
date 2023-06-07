
<p align="center"> <img width="350" src="https://user-images.githubusercontent.com/80123386/244021140-8bfda55c-bb39-424e-bf0e-e7249d7881f0.png"> </p>

# DoppeLLaMA

This is a project to create an LLM powered Discord bot that can impersonate users through fine-tuning with QLoRA.
This bot was designed to locally host an open-source model on the user's PC and not require any paid text-generation services.
Development and testing was done on an Nvidia RTX 3060.

## Local Install (Linux/Mac)

Create a new discord bot, name it "DoppLLaMA", and copy and paste your bot token into the `.env-template` file (See the colab link for instructions on how to set up a new bot). Rename `.env-template` to `.env`

Clone the repo and then run the `setup.sh` and `run.sh` commands


Unfortunately, DoppeLLaMA needs libraries that are not compatible with windows. So Currently, Windows isn't supported.

## Google Colab

https://colab.research.google.com/drive/1TpuEuXC16uIYST938lFHmX95Vb5xqDCe?usp=sharing


<p align="center"> <img width="500" src="https://user-images.githubusercontent.com/80123386/244021164-09b7b045-c008-4d42-96e7-18eb6c593707.png"> </p>
