import discord, os, torch, json, sentencepiece
from discord import app_commands
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, BitsAndBytesConfig
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
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
print("Loading model...")
model_id = "decapoda-research/llama-7b-hf"

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=nf4_config)
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

@tree.command(name="load", description="This command loads a LLM model into the bot")
async def load(interaction: discord.Interaction):
    global model
    global tokenizer

    print("Loading model...")
    await interaction.response.send_message('Loading model...')
    model_id = "kuleshov/llama-7b-4bit"

    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=nf4_config)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(model)
    print(tokenizer)
    await interaction.followup.send("Model loaded")

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
    outputs = model.generate(**inputs, max_new_tokens=150, temperature=1.1)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(result)

    response = interaction.user.display_name + ": " + prompt + "\n" + "Output: " + result
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
                    messages.append({"user": str(message.author), "content": message.content})
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
async def train(interaction: discord.Interaction):
    #TODO: Scrape the server for messages
    #TODO: Format the messages into a training set
    #TODO: Utilizing QLoRA, train the LLM

    global model
    if(model is None):
        print("No model loaded")
        return
    
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    config = LoraConfig(
        r=8, 
        lora_alpha=32, 
        target_modules=["query_key_value"], 
        lora_dropout=0.05, 
        bias="none", 
        task_type="CAUSAL_LM"
    )

    peftModel = get_peft_model(model, config)
    print_trainable_parameters(peftModel)
    


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