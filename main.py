import discord
import logging
import sys
from os.path import join, dirname
from pathlib import Path
from dotenv import load_dotenv, set_key
from discord import app_commands
from discord.ext import commands, tasks
from discord.errors import ClientException
import os
import string
import random
from pydub import AudioSegment
from bark.generation import SAMPLE_RATE, preload_models, codec_decode, generate_coarse, generate_fine, load_codec_model, generate_text_semantic
from encodec.utils import convert_audio
import numpy as np

from IPython.display import Audio
from scipy.io.wavfile import write

from hubert.pre_kmeans_hubert import CustomHubert
from hubert.customtokenizer import CustomTokenizer

from bark.api import generate_audio
from transformers import AutoProcessor, BarkModel


from typing import List
import torchaudio
import torch
from io import BytesIO

from hubert.hubert_manager import HuBERTManager


dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=int(os.environ.get("LOG_LEVEL")),
        datefmt='%Y-%m-%d %H:%M:%S')

logging.getLogger('discord').setLevel(int(os.environ.get("LOG_LEVEL")))
logging.getLogger('discord.client').setLevel(int(os.environ.get("LOG_LEVEL")))
logging.getLogger('discord.gateway').setLevel(int(os.environ.get("LOG_LEVEL")))
logging.getLogger('discord.voice_client').setLevel(int(os.environ.get("LOG_LEVEL")))


class MyClient(discord.Client):
    def __init__(self, *, intents: discord.Intents):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

class FFmpegPCMAudioBytesIO(discord.AudioSource):
    def __init__(self, source, *, executable='ffmpeg', pipe=False, stderr=None, before_options=None, options=None):
        stdin = None if not pipe else source
        args = [executable]
        if isinstance(before_options, str):
            args.extend(shlex.split(before_options))
        args.append('-i')
        args.append('-' if pipe else source)
        args.extend(('-f', 's16le', '-ar', '48000', '-ac', '2', '-loglevel', 'panic'))
        if isinstance(options, str):
            args.extend(shlex.split(options))
        args.append('pipe:1')
        self._process = None
        try:
            self._process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr)
            self._stdout = io.BytesIO(
                self._process.communicate(input=stdin)[0]
            )
        except FileNotFoundError:
            raise discord.ClientException(executable + ' was not found.') from None
        except subprocess.SubprocessError as exc:
            raise discord.ClientException('Popen failed: {0.__class__.__name__}: {0}'.format(exc)) from exc
    def read(self):
        ret = self._stdout.read(Encoder.FRAME_SIZE)
        if len(ret) != Encoder.FRAME_SIZE:
            return b''
        return ret
    def cleanup(self):
        proc = self._process
        if proc is None:
            return
        proc.kill()
        if proc.poll() is None:
            proc.communicate()

        self._process = None

intents = discord.Intents.all()
client = MyClient(intents=intents)


def allowed_wav(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in "wav"

def voice_clone(voice_name, audio_filepath):
  device = 'cuda' # or 'cpu'
  model = load_codec_model(use_gpu=True if device == 'cuda' else False)

  hubert_model = CustomHubert(checkpoint_path='./data/models/hubert/hubert.pt').to(device)
  tokenizer = CustomTokenizer.load_from_checkpoint('./data/models/hubert/model.pth').to(device)


  #audio_filepath = 'data/voices/' + voice_name + '.wav' # the audio you want to clone (under 13 seconds)
  wav, sr = torchaudio.load(audio_filepath)
  wav = convert_audio(wav, sr, model.sample_rate, model.channels)
  wav = wav.to(device)

  semantic_vectors = hubert_model.forward(wav, input_sample_hz=model.sample_rate)
  semantic_tokens = tokenizer.get_token(semantic_vectors)
  
  with torch.no_grad():
      encoded_frames = model.encode(wav.unsqueeze(0))
  codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze() 
  
  
  codes = codes.cpu().numpy()
  
  semantic_tokens = semantic_tokens.cpu().numpy()
  output_path = 'bark/assets/prompts/' + voice_name + '.npz'
  np.savez(output_path, fine_prompt=codes, coarse_prompt=codes[:2, :], semantic_prompt=semantic_tokens)
  preload_models(path="bark/assets/prompts")

def talk(voice_name, text_prompt):
  audio_array = generate_audio(text_prompt, history_prompt=voice_name, text_temp=0.7, waveform_temp=0.7, silent=False)

  bytes_wav = bytes()
  bytes_io = BytesIO(bytes_wav)
  write(bytes_io, SAMPLE_RATE, audio_array)
  result_bytes = bytes_io.read()
  #out = BytesIO()
  #sound = AudioSegment.from_wav(bytes_io)
  #sound.export(out, format='mp3', bitrate="256k")
  #out.seek(0)

  return bytes_io


async def rps_autocomplete(interaction: discord.Interaction, current: str) -> List[app_commands.Choice[str]]:
    choices = {}
    for file in os.listdir(os.environ.get("NPZ_PATH")):
        if file.endswith('.npz'):
            choices [file.split(".")[0]] = file
    choices = [app_commands.Choice(name=choice, value=choice) for choice in choices if current.lower() in choice.lower()][:25]
    return choices

async def connect_bot_by_voice_client(voice_client, channel, guild, member=None):
    try:  
        if (voice_client and not voice_client.is_playing() and voice_client.channel and voice_client.channel.id != channel.id) or (not voice_client or not voice_client.channel):
            if member is not None and member.id is not None:
                if voice_client and voice_client.channel:
                    for memberSearch in voice_client.channel.members:
                        if member.id == memberSearch.id:
                            channel = voice_client.channel
                            break
            perms = channel.permissions_for(channel.guild.me)
            if (perms.administrator or perms.speak):
                if voice_client and voice_client.channel and voice_client.is_connected():
                    await voice_client.disconnect()
                    time.sleep(5)
                await channel.connect()
    except TimeoutError as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logging.error("%s %s %s", exc_type, fname, exc_tb.tb_lineno, exc_info=1)
    except ClientException as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logging.error("%s %s %s", exc_type, fname, exc_tb.tb_lineno, exc_info=1)
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logging.error("%s %s %s", exc_type, fname, exc_tb.tb_lineno, exc_info=1)

async def send_error(e, interaction):
    if isinstance(e, app_commands.CommandOnCooldown):
        await interaction.followup.send("Per favore non spammare.", ephemeral = True)  
    else:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logging.error("%s %s %s", exc_type, fname, exc_tb.tb_lineno, e.args[0])
        await interaction.followup.send("Si é verificato un errore.", ephemeral = True)  

def get_voice_client_by_guildid(voice_clients, guildid):
    for vc in voice_clients:
        if vc.guild.id == guildid:
            return vc
    return None

@client.event
async def on_guild_available(guild):
    try:
        client.tree.copy_global_to(guild=guild)
        await client.tree.sync(guild=guild)
        logging.info(f'Syncing commands to Guild (ID: {guild.id}) (NAME: {guild.name})')
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logging.error("%s %s %s", exc_type, fname, exc_tb.tb_lineno, exc_info=1)

@client.tree.command()
@app_commands.rename(username='username')
@app_commands.describe(username="Il nome dell'user da clonare")
@app_commands.rename(audio='audio')
@app_commands.describe(audio="L'audio da clonare")
@app_commands.checks.cooldown(1, 60.0, key=lambda i: (i.user.id))
async def clone(interaction: discord.Interaction, username: str, audio: discord.Attachment):
    """Clona una voce."""
    try:
        await interaction.response.defer(thinking=True, ephemeral=True)
        if not utils.allowed_wav(audio.filename):
            await interaction.followup.send("Devi caricare un file wav se vuoi clonare una voce.", ephemeral = True)     
        else:
        
            url = get_api_url() + os.environ.get("API_PATH_DATABASE") + "/upload/trainfile/txt"
            form_data = {'chatid': str(currentguildid),
                        'lang': utils.get_guild_language(currentguildid)
                        }
            audiofile = await audio.to_file()
            filepath = os.environ.get("TMP_DIR") + username + ".wav"
            with open(filepath, 'wb') as filewrite:
                filewrite.write(audiofile.fp.getbuffer())
            
            voice.voice_clone(username, filepath)
            if os.path.exists(filepath):
                os.remove(filepath)
            await interaction.followup.send("Voce clonata!\nRicorda che prossimo riavvio del bot tutte le voci clonate verranno eliminate.", ephemeral = True)  
    except Exception as e:
        send_error(interaction, e)

@client.tree.command()
@app_commands.rename(text='text')
@app_commands.describe(text="Il testo da ripetere")
@app_commands.rename(voice='voice')
@app_commands.describe(voice="La voce clonata da usare")
@app_commands.autocomplete(voice=rps_autocomplete)
@app_commands.checks.cooldown(1, 60.0, key=lambda i: (i.user.id))
async def speak(interaction: discord.Interaction, text: str, voice: str):
    """Parla"""
    try:
        await interaction.response.defer(thinking=True, ephemeral=True)    

        voice_client = get_voice_client_by_guildid(client.voice_clients, interaction.guild.id)
        await connect_bot_by_voice_client(voice_client, interaction.user.voice.channel, interaction.guild)
                 
        if not voice_client:
            await interaction.followup.send("Sto entrando nel canale vocale, se qualcosa non dovesse funzionare usa prima /join!", ephemeral = True)  
        else:
            if voice_client.is_playing():
                voice_client.stop()
            
            voice_client.play(FFmpegPCMAudioBytesIO(talk(voice, text), pipe=True), after=lambda e: logging.info("do_play - " + message))
        
            await interaction.followup.send(text, ephemeral = True)  
    except Exception as e:
        send_error(interaction, e)

@client.tree.command()
@app_commands.checks.cooldown(1, 5.0, key=lambda i: (i.user.id))
async def join(interaction: discord.Interaction):
    """Entra in un canale vocale."""
    is_deferred=True
    try:
        await interaction.response.defer(thinking=True, ephemeral=True)
        
        voice_client = get_voice_client_by_guildid(client.voice_clients, interaction.guild.id)
        if voice_client:       
            await voice_client.disconnect()
            time.sleep(5)
        await connect_bot_by_voice_client(voice_client, interaction.user.voice.channel, interaction.guild)

        await interaction.followup.send("Entro nel canale vocale.", ephemeral = True) 
         
    except Exception as e:
        send_error(interaction, e)


@client.tree.command()
@app_commands.checks.cooldown(1, 5.0, key=lambda i: (i.user.id))
async def leave(interaction: discord.Interaction):
    """Esci da un canale vocale"""
    is_deferred=True
    try:
        await interaction.response.defer(thinking=True, ephemeral=True)
        check_permissions(interaction)
        
        voice_client = get_voice_client_by_guildid(client.voice_clients, interaction.guild.id)
        if voice_client:       
            await voice_client.disconnect()     
            await interaction.followup.send("Esco dal canale vocale.", ephemeral = True)
        else:
            await interaction.followup.send("Non sono collegato a nessun canale vocale.", ephemeral = True)       
         
    except Exception as e:
        send_error(interaction, e) 
        
@speak.error
@clone.error
@join.error
@leave.error
async def on_generic_error(interaction: discord.Interaction, e: app_commands.AppCommandError):
    send_error(interaction, e)

logging.info("Starting Models Loading...")

hubert_manager = HuBERTManager()
hubert_manager.make_sure_hubert_installed()
hubert_manager.make_sure_tokenizer_installed()

preload_models(path=os.environ.get("NPZ_PATH"))


logging.info("Starting Discord Client...")

client.run(os.environ.get("BOT_TOKEN"))