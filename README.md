# DoppeLLaMA

This is a WIP project to create a LLM powered Discord bot that can impersonate users through fine-tuning with QLoRA.
This bot was designed to locally host an open-source model on the user's PC and not require any paid text-generation services.
Development and testing was done on an Nvidia RTX 3060.

## Environment Setup

To set up the required Conda environment to run this project, follow these steps:

1. Clone the repository:
  `git clone https://github.com/Urist-Mc-Urist/DoppeLLaMA.git`

2. Navigate to the project directory:
  `cd doppellama`

3. Create a Conda environment from the environment.yml file:
  `conda env create --file environment.yml`

4. Activate the newly created environment:
  `conda activate DL`

## Creating a bot instance

This is only a locally hosted bot. You will have to create your own Discord bot through the developer portal
and then run the code in this repo.

1. Set up your own discord bot:</br>
  Go to https://discord.com/developers/applications</br>
  Select "New Application"</br>
  Navigate to "Bot"</br>
  Create and save a new Token</br>
  Place the token in `.env-template`</br>
  
2. Add the bot to a server</br>
  Under the left side Settings panel select OAuth2 > URL Generator</br>
  Select `bot` from the options</br>
  Select `Send Messages` and `Read Message History` from the second menu</br>
  Copy the URL, paste into Discord, and add the bot to your server of choice.
