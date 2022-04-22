from ast import alias
from code import interact
import discord
import os
from dotenv import load_dotenv
from discord.ext import commands
import requests

load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
ctr:int =0
prefix = "."
bot = commands.Bot(prefix)

@bot.event
async def on_command_error(ctx,error):
    pass

@bot.event
async def on_ready():
    activity = discord.Activity(name='my activity', type=discord.ActivityType.watching)
    client = discord.Client(activity=activity)
    
    guild_count = 0

    for guild in bot.guilds:
        print(f"- {guild.id} (name: {guild.name})")             # PRINT THE SERVER'S ID AND NAME.
        guild_count = guild_count + 1

    print("SampleDiscordBot is in " + str(guild_count) + " guilds.")

    

@bot.command()
async def hi(ctx):
    await ctx.send("Hey Buddy")
@bot.event
async def on_message(message):
    PARAMS = {'address':message.content}
    
    r = requests.get(url = 'http://127.0.0.1:5000/predict', params = PARAMS)
    print(r.text)
    resp=int(r.text)
    if resp == 1:
        await message.channel.send("Please adhere to the rules, Do not be hateful towards each other")
    await bot.process_commands(message)    

@bot.command(pass_context = True, alias = "purge")
async def clear(ctx,ammount):
    if isinstance(ammount,int): await ctx.channel.purge(limit=ammount)
    elif ammount == "all": await ctx.channel.purge()

@clear.error
async def clear_error_handler(ctx,error):
    if isinstance(error, commands.MissingRequiredArgument):
        if error.param.name == "ammount":
            await ctx.send("Enter the ammount of messages to be deleted")



bot.run(DISCORD_TOKEN)

