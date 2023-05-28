import discord
from discord import app_commands
import os
from dotenv import load_dotenv

#load API key and set intents
load_dotenv()
api_key = os.getenv('DISCORD_API_KEY')
intents = discord.Intents.default()
intents.message_content = True
permissions_integer = 67584 #from Discord Dev portal

#initialize bot
bot = discord.Client(intents=intents, permissions=discord.Permissions(permissions_integer))
tree = app_commands.CommandTree(bot)

#bot events
@bot.event
async def on_ready():
    await tree.sync()
    print(f"Bot is ready. Connected to {len(bot.guilds)} guild(s).")

@tree.command(name="test", description="Test command")
async def ping(interaction: discord.Interaction):
    await interaction.response.send_message('Hello World')

@tree.command(name="load", description="This command loads a LLM model into the bot")
async def load(interaction: discord.Interaction):
    pass
    #TODO: Load the LLM

@tree.command(name="unload", description="This command unloads the LLM from memory")
async def unload(interaction: discord.Interaction):
    pass
    #TODO: unload the LLM

@tree.command(name="train", description="Train an LLM model on the message content of the server")
async def train(interaction: discord.Interaction):
    pass
    #TODO: Scrape the server for messages
    #TODO: Format the messages into a training set
    #TODO: Utilizing QLoRA, train the LLM

bot.run(api_key)