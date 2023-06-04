import discord, os, torch, json, transformers, asyncio, random
from discord import app_commands
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM,  LlamaTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model

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
lora_loaded = False
currently_training = False
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
print("Loading model...")
model_id = "decapoda-research/llama-7b-hf"

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=nf4_config, device_map={"":0})

tokenizer = LlamaTokenizer.from_pretrained(model_id)
print("Base model loaded")

#=============== Bot Events and Commands =================
@bot.event
async def on_ready():
    await tree.sync()
    print(f"Bot is ready. Connected to {len(bot.guilds)} guild(s).")


@tree.command(name="load", description="Load this server's trained model into the bot")
async def load(interaction: discord.Interaction):
    global model
    global tokenizer
    global lora_loaded
    global currently_training
    base_model = model

    if currently_training:
        await interaction.response.send_message("Error: Currently training")
        return


    print("Attempting to load model...")
    await interaction.response.send_message("Attempting to load model...")
    
    #create new thread to load model, loading the LoRAs is quick so probably not needed, but good practice
    with ThreadPoolExecutor() as executor:
        lora_path = f"./models/{interaction.guild_id}"
        if(lora_path):
            await interaction.followup.send("Found model")
            loop = asyncio.get_event_loop()
            model = await loop.run_in_executor(executor, load_model, lora_path, base_model)
            lora_loaded = True
            await interaction.followup.send("Model successfully loaded")
        else:
            await interaction.followup.send("No model found for current guild")

def load_model(lora_path, base_model):
    peft_model_id = lora_path
    model = PeftModel.from_pretrained(base_model, peft_model_id)
    return model


@tree.command(name="impersonate", description="Impersonate a user. user: 'user#0434', tokens <= 400")
async def basic_prompt(interaction: discord.Interaction, prompt: str, user: str, tokens: int):
    global model
    global tokenizer
    global currently_training

    if currently_training:
        await interaction.response.send_message("Error: Currently training")
        return

    if tokens > 400 :
        await interaction.response.send_message('Too many tokens. (Must be <= 400)')
        return

    await interaction.response.send_message('Generating response...')

    system_message = "Below is an interaction between two users in a chatroom::\n\n"
    example1 = "SlimShady#4875: What should I make for dinner? I'm thinking about making some fried chicken\n\nBigMike#0232: You could always order a pizza or something\n\n"
    example2 = "ExampleMan#3424: I'm going to hop on for some games, anyone want to join?\n\nZoomerBoomer#5995: Yeah I'm down, what did you want to play?\n\n"
    #example3 = "TechGuru#1234: Hey everyone, I'm facing a frustrating issue with my computer. It's been freezing constantly, and I'm not sure what's causing it. I've tried running a virus scan and clearing up some disk space, but nothing seems to be working. Any suggestions or troubleshooting tips would be greatly appreciated!\n\nCodeMaster#9876: Hey TechGuru#1234, sorry to hear about your computer troubles. Freezing issues can be caused by various factors. Have you checked the CPU temperature? Overheating can lead to system freezes. You can use monitoring software to keep an eye on the temperature. Additionally, updating your drivers and checking for any hardware conflicts might also help. Let's troubleshoot this together!"
    #example4 = "BookLover#2341: I've been on a quest for an enthralling fantasy novel that will transport me to magical realms and captivate my imagination. Any recommendations? I crave epic adventures filled with heroic characters and intricate plots.\n\nStoryteller#9987: Absolutely! I highly recommend diving into 'The Name of the Wind' by Patrick Rothfuss. It's a spellbinding tale of a legendary hero's journey, brimming with breathtaking world-building and unforgettable moments. Prepare to lose yourself in its pages!"
    #example5 = "UserA#1231: What are you guys doing?\n\nUserb#0232: Just haning out, what about you?\n\n"
    system_prompt = system_message + example1 + system_message + example2 + system_message #+ example3 + system_message + example4 + system_message + example5 + system_message
    model_input = system_prompt + str(interaction.user) + ": " + prompt + "\n\n" + user + ":"
    inputs = tokenizer(model_input, return_tensors="pt").to('cuda')

    def generate_wrapper(model, tokens):
        seed = random.randint(0, 10000)
        torch.manual_seed(seed)

        with torch.no_grad():
            return model.generate(
                **inputs, 
                max_new_tokens=tokens, 
                temperature=0.9,
                top_p=0.72,
                top_k=0,
                repetition_penalty=1.1
                )

    with ThreadPoolExecutor() as executor:
        loop = asyncio.get_event_loop()
        outputs = await loop.run_in_executor(executor, generate_wrapper, model, tokens)

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(result)

    #remove additional system prompt from response
    discard = len(model_input)
    generated_text = result[discard:].splitlines()[0]
    response = str(interaction.user) + ": " + prompt + "\n\n" + user + ":" + generated_text
    await interaction.followup.send(response)


@tree.command(name="scrape", description="scrapes messages from server")
async def scrape(interaction: discord.Interaction, limit: int, channel: str):
    global currently_training
    if currently_training:
        await interaction.response.send_message("Error: Currently training")
        return

    await interaction.response.send_message('Beginning scrape...')
    guild = interaction.guild
    print(f"Scanning {guild.name}")

    #Find text channel with specified name
    target_channel = None
    for text_channel in guild.text_channels:
        if text_channel.name == channel:
            #set target_channel if found
            target_channel = text_channel

    #check if specified channel actually exists
    if target_channel is None:
        await interaction.followup.send(f"Text channel '{channel}' not found")
        return

    messages = []
    try:
        async for message in target_channel.history(limit=limit):  # Get the message history for the channel
            #remove empty messages and links
            if(message.content and not message.content.startswith('http') and not message.content.startswith('https')):
                print(f"Got: {str(message.author)}: {message.content}")
                messages.append({"message": str(message.author) + ": " + message.content})

    except Exception as e:
        print(f"Could not fetch history for channel {target_channel.name}, possibly missing permissions.")

    print(f"Found {len(messages)} messages")
    
    #Save the message list as a JSON file in ./messages/{guild.id}
    os.makedirs(f'./messages/{guild.id}', exist_ok=True)
    with open(f'./messages/{guild.id}/messages.json', 'w') as outfile:
        json.dump(messages, outfile)

    await interaction.followup.send('Scrape complete')


@tree.command(name="train", description="Train an LLM model on the message content of the server")
async def train(interaction: discord.Interaction, training_steps: int):
    global model
    global tokenizer
    global lora_loaded
    global currently_training

    if currently_training:
        await interaction.response.send_message("Error: Currently training")
        return

    if lora_loaded:
        await interaction.response.send_message('A model has already been loaded for inference\n\n If you want to train a model on this server, reset the bot call "/train" without calling "/load"')
        return
    
    await interaction.response.send_message('Beginning training...')
    currently_training = True

    guild = interaction.guild

    #Load training data (messages)
    filename = f"./messages/{guild.id}/messages.json"
    if os.path.exists(filename):
        with open(filename, "r") as file:
            #convert json file into transformers dataset and tokenize messages
            data = load_dataset("json", data_files=filename)
            print(data["train"])
            data = data.map(lambda samples: tokenizer(samples["message"]), batched=True)
            print("Training data loaded")
    else:
        interaction.followup.send("Error: No training data. '/scrape' server first")
        return
    
    #Create a PEFT model to train
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

    #Create new thread to run training on
    with ThreadPoolExecutor() as executor:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(executor, train_model, model, tokenizer, data, guild.id, training_steps)
    
    currently_training = False


def train_model(model, tokenizer, data_set, guild_id, training_steps):

    #LLaMA tokenizer needs this for some reason
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    #specify the training args
    #Still not sure what the optimal steps per messages is. 1 step per message seems to work
    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=10,
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
    model.save_pretrained(f"./models/{guild_id}")

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