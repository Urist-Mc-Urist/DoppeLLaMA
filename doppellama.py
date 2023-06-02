import discord, os, torch, json, transformers, datasets, asyncio
from discord import app_commands
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, BitsAndBytesConfig
from datasets import Dataset, load_dataset
from concurrent.futures import ThreadPoolExecutor

from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

#load API key and set intents
load_dotenv()
api_key = os.getenv('DISCORD_API_KEY')
intents = discord.Intents.default()
intents.message_content = True
permissions_integer = 67584 #from Discord Dev portal

#initialize bot
bot = discord.Client(intents=intents, permissions=discord.Permissions(permissions_integer))
tree = app_commands.CommandTree(bot)

#declare model and tokenizer
model = None
tokenizer = None
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
print("Loading model...")
model_id = "decapoda-research/llama-7b-hf"

#model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=nf4_config, device_map={"":0})
model = AutoModelForCausalLM.from_pretrained("./models/1099579232839024710", quantization_config=nf4_config, device_map={"":0})
tokenizer = LlamaTokenizer.from_pretrained(model_id)
print(model)
print(tokenizer)

#bot events
@bot.event
async def on_ready():
    await tree.sync()
    print(f"Bot is ready. Connected to {len(bot.guilds)} guild(s).")


@tree.command(name="test", description="Test command")
async def ping(interaction: discord.Interaction):
    print("Test command received")
    await interaction.response.send_message('Hello World')


@tree.command(name="load", description="Load this server's trained model into the bot")
async def load(interaction: discord.Interaction):
    global model
    global tokenizer
    global nf4_config
    print(nf4_config)

    print("Attempting to load model...")
    await interaction.response.send_message("Attempting to load model...")
    
    with ThreadPoolExecutor() as executor:
        loop = asyncio.get_event_loop()
        model_path = f"./models/{interaction.guild_id}"
        if(model_path):
            model = None
            await interaction.followup.send("Found model")
            model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=nf4_config, device_map={"":0})
            #model = await loop.run_in_executor(executor, load_model, model_path, nf4_config, {"": 0})
            await interaction.followup.send("Model successfully loaded")
        else:
            await interaction.followup.send("No model found for current guild")

def load_model(model_path, quantization_config, device_map):
    return AutoModelForCausalLM.from_pretrained(model_path, quantization_config=quantization_config, device_map=device_map)


@tree.command(name="basic_prompt", description="Query the LLM with a prompt")
async def basic_prompt(interaction: discord.Interaction, prompt: str):
    global model
    global tokenizer

    if(model is None):
        await interaction.response.send_message("No model loaded")
        return

    await interaction.response.send_message('Generating response...')

    device = "cuda:0"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs, 
        max_new_tokens=150, 
        temperature=0.72,
        top_p=0.73,
        top_k=0,
        repetition_penalty=1.2
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(result)

    response = interaction.user.display_name + ": " + prompt + "\n\n" + "Output: " + result
    await interaction.followup.send(response)


@tree.command(name="scrape", description="scrapes messages from server")
async def dataset(interaction: discord.Interaction, limit: int):
    await interaction.response.send_message('Beginning scrape...')
    guild = interaction.guild
    print(f"Scanning {guild.name}")
    messages = []
    for channel in guild.text_channels:  # Iterate over every text channel
        print(f"Scanning {channel.name}...")
        try:
            async for message in channel.history(limit=limit):  # Get the message history for the channel
                if(message.content and not message.content.startswith('http') and not message.content.startswith('https')):
                    print(f"Got: {str(message.author)}: {message.content}")
                    messages.append({"message": str(message.author) + message.content})
        except Exception as e:
            print(f"Could not fetch history for channel {channel.name}, possibly missing permissions.")
    
    os.makedirs(f'./messages/{guild.id}', exist_ok=True)
    with open(f'./messages/{guild.id}/messages.json', 'w') as outfile:
        json.dump(messages, outfile)

    await interaction.followup.send('Scrape complete')


@tree.command(name="unload", description="This command unloads the LLM from memory")
async def unload(interaction: discord.Interaction):
    pass
    #TODO: unload the LLM


@tree.command(name="train", description="Train an LLM model on the message content of the server")
async def train(interaction: discord.Interaction, training_steps: int):
    global model
    global tokenizer
    if(model is None):
        print("No model loaded")
        return
    
    await interaction.response.send_message('Beginning training...')
    guild = interaction.guild

    filename = f"./messages/{guild.id}/messages.json"
    if os.path.exists(filename):
        with open(filename, "r") as file:
            data = load_dataset("json", data_files=filename)
            print(data["train"])
            data = data.map(lambda samples: tokenizer(samples["message"]), batched=True)
            print("Training data loaded")
    else:
        interaction.followup.send_message("Error: No training data. '/scrape' server first")
        return
    
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    config = LoraConfig(
        r=8, 
        lora_alpha=32, 
        lora_dropout=0.05, 
        bias="none", 
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)

    with ThreadPoolExecutor() as executor:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(executor, train_model, model, tokenizer, data, guild.id, training_steps)
    
    await interaction.followup.send("Training complete")


def train_model(model, tokenizer, data_set, guild_id, training_steps):
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=training_steps,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir=f"./models/{guild_id}",
        optim="paged_adamw_8bit"
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=data_set["train"],
        args=training_args,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    print("Saving model")
    trainer.save_model(output_dir=f"./models/{guild_id}")

    # Save the model configuration
    print("Saving model configuration")
    config = model.config
    config.save_pretrained(f"./models/{guild_id}")

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

bot.run(api_key)