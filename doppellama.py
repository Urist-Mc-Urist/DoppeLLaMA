import discord, os, torch, json, transformers, asyncio, random
from discord import app_commands
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, LlamaTokenizer, BitsAndBytesConfig
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from datasets import Dataset
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
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))
print("Base model loaded")

#=============== Bot Events and Commands =================
@bot.event
async def on_ready():
    await tree.sync()
    print(f"Bot is ready. Connected to {len(bot.guilds)} guild(s).")


#This method needs to be overly complicated because we're loading a PEFT model instead of a base CausalLM. 
#Ideally we'd just be able to just specify the load_best_model_at_end but that's not set up to create a PEFT model
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
    await interaction.response.send_message(content="Attempting to load model...", delete_after=15)

    # Create new thread to load model, loading the LoRAs is quick so probably not needed, but good practice
    lora_path = f"./models/{interaction.guild_id}"
    
    best_checkpoint_path = None
    lowest_eval_loss = float('inf')  # start with infinity so that any real loss will be lower
    
    # iterate over all checkpoint directories
    for entry in os.scandir(lora_path):
        if entry.is_dir() and "checkpoint" in entry.name:
            checkpoint_path = os.path.join(lora_path, entry.name)
            print(f"checking {checkpoint_path}")
            trainer_state_file = os.path.join(checkpoint_path, "trainer_state.json")

            # read trainer_state.json
            with open(trainer_state_file, 'r') as file:
                trainer_state = json.load(file)

            # get eval_loss of last log
            last_log = trainer_state["log_history"][-1]  # last log
            if "eval_loss" in last_log:
                eval_loss = last_log["eval_loss"]

                if eval_loss < lowest_eval_loss:
                    print("Found better model")
                    lowest_eval_loss = eval_loss
                    best_checkpoint_path = checkpoint_path

    if best_checkpoint_path:
        lora_model_path = os.path.join(best_checkpoint_path, "adapter_model")
        print(f"Loading model from {lora_model_path}")
        await interaction.edit_original_response(content=f"Loading model from {lora_model_path}")

        with ThreadPoolExecutor() as executor:
            loop = asyncio.get_event_loop()
            model = await loop.run_in_executor(executor, load_model, lora_model_path, base_model)

        model.resize_token_embeddings(len(tokenizer))
        lora_loaded = True
        await interaction.edit_original_response(content="Model successfully loaded")
    else:
        await interaction.edit_original_response(content="No model found for current guild")

def load_model(lora_path, base_model):
    peft_model_id = lora_path
    model = PeftModel.from_pretrained(base_model, peft_model_id)
    return model

@tree.command(name="impersonate", description="Generate a response based on most recent chat history")
async def impersonate(interaction: discord.Interaction, user: str):
    global model
    global tokenizer
    global currently_training

    if currently_training:
        await interaction.response.send_message(content="Error: Currently training", delete_after=15)
        return
    

    messages = []
    async for message in interaction.channel.history(limit=10):  # Get the message history for the channel
        content = message.content
        username = message.author.display_name
        #remove empty messages and links
        if(message.content):
            print(f"Got: {username}: {content}")
            if "DoppeLLaMA" in username:
                messages.insert(0, message.content)
            else:
                messages.insert(0, (username + ": " + content))
    
    await interaction.response.defer()

    system_prompt = ""
    for message in messages:
        system_prompt += message + "\n\n"
    
    system_prompt +=  user + ":"

    inputs = tokenizer(system_prompt, return_tensors="pt").to('cuda')

    with ThreadPoolExecutor() as executor:
        loop = asyncio.get_event_loop()
        outputs = await loop.run_in_executor(executor, generate_wrapper, model, inputs)

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(result)

    #remove additional system prompt from response
    generated_text = result[len(system_prompt):]

    response = user + ":" + generated_text
    await interaction.edit_original_response(content=response)


def generate_wrapper(model, inputs):
        global tokenizer
        seed = random.randint(0, 10000)
        torch.manual_seed(seed)

        eos_token = "\n\n"
        eos_token_id = tokenizer.encode(eos_token)

        with torch.no_grad():
            return model.generate(
                #You may need to play around with these values to get better generations, particularly repetition_penalty
                **inputs, 
                max_new_tokens=200, 
                temperature=0.9,
                top_p=0.72,
                top_k=40,
                repetition_penalty=1.15, 
                early_stopping=True,
                eos_token_id=eos_token_id
                )


@tree.command(name="scrape", description="scrapes messages from server")
async def scrape(interaction: discord.Interaction, limit: int, channel: str):
    global currently_training
    if currently_training:
        await interaction.response.send_message("Error: Currently training")
        return

    await interaction.response.defer()

    guild = interaction.guild
    print(f"Scanning {channel} in {guild.name}")

    #Find text channel with specified name
    target_channel = None
    for text_channel in guild.text_channels:
        if text_channel.name == channel:
            #set target_channel if found
            target_channel = text_channel

    #check if specified channel actually exists
    if target_channel is None:
        await interaction.edit_original_response(content=f"Text channel '{channel}' not found")
        return

    messages = []
    try:
        count = 0
        message_block = ""
        async for message in target_channel.history(limit=limit):  # Get the message history for the channel
            display_name = message.author.display_name 
            content = message.content

            #remove empty messages and links
            if(content is None or content == "" or content.startswith('http') or content.startswith('https')):
                continue

            #Messages are trimmed to 255 chars saved in batches of 6 to capture context and interaction
            print(f"Got: {str(display_name)}: {message.content}")
            #list is not full
            if count < 6:
                message_block = message_block + display_name + ": " + content[:255] + "\n\n"
                count += 1
            #list is full
            else:
                messages.append({"message": message_block})

                count = 1
                message_block = display_name + ": " + content[:255] + "\n\n"

    except Exception as e:
        print(e)
        print(f"Could not fetch history for channel {target_channel.name}, possibly missing permissions.")
        await interaction.edit_original_response(content='Error during scrape, possibly missing permissions?')
        return

    print(f"Collected {len(messages)} message sets")
    await interaction.edit_original_response(content=f"Collected {len(messages)} message sets")
    
    #Save the message list as a JSON file in ./messages/{guild.id}
    os.makedirs(f'./messages/{guild.id}', exist_ok=True)
    with open(f'./messages/{guild.id}/messages.json', 'w') as outfile:
        json.dump(messages, outfile)

    print(f"Scrape complete\nCollected {len(messages)} message sets")
    await interaction.edit_original_response(content=f"Scrape complete\nCollected {len(messages)} message sets")


@tree.command(name="train", description="Train an LLM model on the message content of the server")
async def train(interaction: discord.Interaction):
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
    
    await interaction.response.send_message('Beginning training...\nDiscord interactions time out after 15 minutes, please check the bot logs for updates.')
    currently_training = True

    guild = interaction.guild

    #Load training data (messages)
    filename = f"./messages/{guild.id}/messages.json"
    if os.path.exists(filename):
        json_data = json.load(open(filename, "r"))
        messages = [item['message'] for item in json_data]
        data_dict = {"text": messages}
        dataset = Dataset.from_dict(data_dict)
        print("Total data:", dataset)

        tokenized_dataset = dataset.map(lambda samples: tokenizer(samples["text"]))

        tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.15)
        dataset_train = tokenized_dataset["train"]
        dataset_eval = tokenized_dataset["test"]

        print("Train data:", dataset_train)
        print("Evaluation data:", dataset_eval)

        print("Training and evaluation data loaded")
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
        await loop.run_in_executor(executor, train_model, model, tokenizer, dataset_train, dataset_eval, guild.id)
    
    currently_training = False
    print("=============  Training complete  =============")


def train_model(model, tokenizer, training_data, test_data, guild_id):

    training_args = transformers.TrainingArguments(
        #Any higher of a batch size would crash my RTX 3060
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=5,
        output_dir=f"./models/{guild_id}",
        optim="paged_adamw_8bit",
        metric_for_best_model="loss",
        greater_is_better=False,
        evaluation_strategy="steps",
        eval_steps=5,
        save_steps=5 
    )

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=training_data,
        eval_dataset=test_data,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[SavePeftModelCallback]
    )

    trainer.train()

    #Run a final evaluation and save trainer_state.json into the final checkpoint folder.
    #As far as I can tell this can't be done in the callback because you don't have access to `trainer` directly
    trainer.evaluate()
    trainer.state.save_to_json(f"./models/{guild_id}/checkpoint-{trainer.state.global_step}/trainer_state.json")

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


class SavePeftModelCallback(transformers.TrainerCallback):
    def on_save(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )       

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        self.remove_uneeded_files(checkpoint_folder) #remove to keep base model (4 Gb) and other files in each checkpoint

        return control
    
    def on_train_end(
        self,
        args: transformers.TrainingArguments,
        state: transformers.TrainerState,
        control: transformers.TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        self.remove_uneeded_files(checkpoint_folder) #remove to keep base model (4 Gb) and other files in each checkpoint
        return control
    
    def remove_uneeded_files(self, checkpoint_folder):
        uneeded_files = ["pytorch_model.bin", "optimizer.pt", "rng_state.pth", "scheduler.pt", "training_args.bin"]

        for file_name in uneeded_files:
            to_remove = os.path.join(checkpoint_folder, file_name)
            if os.path.exists(to_remove):
                os.remove(to_remove)


bot.run(api_key)