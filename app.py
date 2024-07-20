import subprocess
import re
import streamlit as st
import numpy as np
import random
import time
from PIL import Image
import os
import mimetypes
from supabase import create_client, Client, StorageException
from io import StringIO, BytesIO
from tempfile import NamedTemporaryFile
import json
import requests
import unicodedata
import logging
import base64
from openai import OpenAI
import streamlit_extras
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.row import row
from pytube import YouTube
from groq_whisper import split_audio,process_files_concurrently
from subtitle_translator import wrap_translate

# set logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# clear logger if exists
if logger.hasHandlers():
	logger.handlers.clear()

# create a console logger
console_handler = logging.StreamHandler()
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)



# check if image
def is_image(file_path):
	try:
		Image.open(file_path)
		return True
	except IOError:
		return False

def get_supabase_client():
	url = st.secrets['supabase_url']
	key = st.secrets['supabase_key']
	supabase = create_client(url, key)
	return supabase

# insert data to database
def supabase_insert_message(table,message):
	supabase = get_supabase_client()
	data, count = supabase.table(table).insert(message).execute()

def supabase_insert_user(name,user_name,profile,picture,oauth_token,email):
	supabase = get_supabase_client()
	data, count = supabase.table('transcript_users').insert({"name":name,"user_name":user_name,"profile":profile,"picture":picture,"oauth_token":oauth_token,"email":email}).execute()


def supabase_fetch_user(user_name):
	supabase = get_supabase_client()
	data,count = supabase.table('transcript_users').select("*").eq('user_name',user_name).execute()
	return data

def update_user_by_email(email,k,v):
	supabase = get_supabase_client()
	data, count = supabase.table('transcript_users').update({k: v}).eq('email', email).execute()
	return data

def supabase_fetch_user_by_email(email):
	supabase = get_supabase_client()
	data,count = supabase.table('transcript_users').select("*").eq('email',email).execute()
	return data

def supabase_fetch_kofi_by_email(email):
	supabase = get_supabase_client()
	data,count = supabase.table('kofi_donation').select("*").eq('email',email).execute()
	return data

# check if file already exists
def check_supabase_file_exists(file_path,bucket_name):
	supabase = get_supabase_client()
	supabase_storage_ls = supabase.storage.from_(bucket_name).list()
	
	if any(file["name"] == os.path.basename(file_path) for file in supabase_storage_ls):
		return True
	else:
		return False


# user unicodedata to remove characters that are not ASCII
def remove_non_ascii(text):
	return ''.join(c for c in text if ord(c) < 128)

def upload_file_to_supabase_storage(file_path):
	base_name = remove_non_ascii(os.path.basename(file_path)).replace(' ', '_')
	path_on_supastorage = os.path.splitext(base_name)[0] + '_' + str(round(time.time())//6000)  + os.path.splitext(base_name)[1]
	mime_type, _ = mimetypes.guess_type(base_name)
	
	supabase = get_supabase_client()
	bucket_name = st.secrets["bucket_name"]

	try:
		if check_supabase_file_exists(path_on_supastorage,bucket_name):
			public_url = supabase.storage.from_(bucket_name).get_public_url(path_on_supastorage)
		else:
			supabase.storage.from_(bucket_name).upload(file=file_path, path=path_on_supastorage, file_options={"content-type": mime_type})
			public_url = supabase.storage.from_(bucket_name).get_public_url(path_on_supastorage)
	except StorageException as e:
		print("StorageException:", e)
		raise
	return public_url


def update_user_msg_pv(email):
	user_data = supabase_fetch_user_by_email(email)
	msg_pv = user_data[1][0]["msg_pv"] if user_data[1][0]["msg_pv"] else 0
	msg_pv += 1
	update_data = update_user_by_email(email,'msg_pv',msg_pv)
	logger.debug(f"user msg_pv update: {email} ---> {msg_pv}")
	return update_data

def is_user_valid(email):
	donation_data = supabase_fetch_kofi_by_email(email)
	user_data = supabase_fetch_user_by_email(email)
	msg_pv = user_data[1][0]["msg_pv"] if user_data[1][0]["msg_pv"] else 0

	if email in st.secrets['user_white_list'].split(','):
		logger.debug("white list user")
		return True
	elif donation_data[1]:
		logger.debug("donation user")
		return True
	elif msg_pv < int(st.secrets["free_quota"]):
		logger.debug("free user less than free quota")
		return True
	else:
		return False

def is_donation(email):
	donation_data = supabase_fetch_kofi_by_email(email)
	if email in st.secrets['user_white_list'].split(','):
		logger.debug("white list user")
		return True
	if donation_data[1]:
		return True
	else:
		return False


def save_kg_json():

	# save kaggle.json
	kaggle_json = {
	"username":st.secrets["kaggle_username"],
	"key":st.secrets["kaggle_api_key"]
	}

	kg_json_dir = os.path.expanduser('~/.kaggle')

	mkdir_kaggle_json = subprocess.run(["mkdir","-p",kg_json_dir],check=True)

	# save kaggle json file
	with open(f'{kg_json_dir}/kaggle.json','w') as f:
		json.dump(kaggle_json,f)

	chmod = subprocess.run(["chmod","600",os.path.join(kg_json_dir, 'kaggle.json')],check=True)

def set_notebook_dir(kg_notebook_dir):
	# remove kg_notebook dir if exists
	rm_kg_notebook = subprocess.run(["rm","-rf",kg_notebook_dir],check=True)

	# create kg_notebook dir 
	mkdir_kg_notebook = subprocess.run(["mkdir","-p",kg_notebook_dir],check=True)

def pull_and_run_notebook(notebook,kg_notebook_dir):
	with notebook_pull_spinner_placeholder:
		with st.spinner("Processing..."):
			# pull notebook code and metadata
			pull = subprocess.run(["kaggle","kernels","pull",notebook,"-p",kg_notebook_dir,"-m"],check=True)
			# check notebook metadata	 
			with open(os.path.join(kg_notebook_dir,'kernel-metadata.json')) as f:
				kg_metadata = json.load(f)
			
			# push notebook
			kg_push = subprocess.run(["kaggle", "kernels", "push", "-p", kg_notebook_dir], check=True)
			return kg_metadata


def check_kernel_status_youtube(notebook, interval=5):
	with notebook_running_spinner_placeholder:
		# 无限循环，直到状态变为"complete"
		with st.spinner("Downloading video..."):
			while True:
				result = subprocess.run(["kaggle", "kernels", "status", notebook], capture_output=True, text=True)
				stdout = result.stdout
				# 提取状态值
				status = re.findall(r'status "(\w+)"',stdout)[0]
				st.session_state.youtube_notebook_status = status
				logger.info(f"The current status is: {status}")
				# 检查状态是否为"complete"
				if status == "complete":
					logger.info("The notebook has finished running.")
					break
				elif status == 'error':
					logger.info("Something wrong, please check the notebook status.")
					break
				else:
					# logger.info("The notebook is still running. Checking again in 5 seconds...")
					time.sleep(interval)  # 等待5秒

def check_kernel_status_transcript(notebook, interval=5):
	# 预估总时长
	estimate_time = 40 + st.session_state.audio_length / 15
	elapsed_time = 0
	with notebook_running_spinner_placeholder:
		with st.spinner("WhisperFlow is now actively transcribing your audio/video."):
			# 初始化进度条
			progress_text = "Transcribe progress"
			my_bar = st.progress(0, text=progress_text)
			# 无限循环，直到状态变为"complete"
			while True:
				result = subprocess.run(["kaggle", "kernels", "status", notebook], capture_output=True, text=True)
				stdout = result.stdout
				# 提取状态值
				status = re.findall(r'status "(\w+)"',stdout)[0]
				st.session_state.notebook_status = status
				logger.info(f"The current status is: {status}")

				progress = min(elapsed_time / estimate_time, 0.99)  # 计算进度百分比，最大为0.99
				my_bar.progress(int(progress * 100), text=progress_text)
				
				# 检查状态是否为"complete"
				if status == "complete":
					logger.info("The notebook has finished running.")
					my_bar.progress(100, text="Transcription completed！")
					break
				elif status == 'error':
					logger.info("Something wrong, please check the notebook status.")
					break
				elif status == 'running':
					elapsed_time += 5
					# logger.info("The notebook is still running. Checking again in 5 seconds...")
					time.sleep(interval)  # 等待5秒


def save_output(notebook,kg_notebook_output_dir):
	with notebook_save_output_spinner_placeholder:
		with st.spinner("Processing... "):
			# remove kg_notebook dir if exists
			kg_rm_output = subprocess.run(["rm","-rf",kg_notebook_output_dir],check=True)
			# create output dir
			kg_mkdir_output = subprocess.run(["mkdir", "-p", kg_notebook_output_dir], check=True)
			# save output
			kg_save_output = subprocess.run(["kaggle", "kernels", "output", notebook, "-p", kg_notebook_output_dir], check=True)

def kg_notebook_run_youtube(notebook,kg_notebook_dir,kg_notebook_output_dir):	
	set_notebook_dir(kg_notebook_dir)
	
	pull_and_run_notebook(notebook,kg_notebook_dir)
	check_kernel_status_youtube(notebook, interval=5)
	if st.session_state.youtube_notebook_status == "complete":
		save_output(notebook,kg_notebook_output_dir)
		st.session_state.youtube_notebook_output = True
		logger.info('output file saved success!')
	else:
		logger.error("opps! something wrong")

def kg_notebook_run_with_transcript(notebook,kg_notebook_dir,kg_notebook_output_dir):	
	set_notebook_dir(kg_notebook_dir)
	
	pull_and_run_notebook(notebook,kg_notebook_dir)
	check_kernel_status_transcript(notebook, interval=5)
	if st.session_state.notebook_status == "complete":
		save_output(notebook,kg_notebook_output_dir)
		st.session_state.notebook_output = True
		logger.info('output file saved success!')
	else:
		logger.error("opps! something wrong")




def check_dataset_status(dataset):
	while True:
		# get dataset creation status
		dataset_status = subprocess.run(["kaggle","datasets","status",dataset],capture_output=True, text=True)
		if dataset_status.stdout == 'ready':
			print("New dataset is ready")
			break
		else:
			print("New dataset is still updating...")
			time.sleep(5)

def update_transcript_audio(dataset,audio_file,kg_notebook_input_data_dir):
	with notebook_data_spinner_placeholder:
		with st.spinner("Preparing your audio/video data for transcription."):
			# remove if exists
			rm_dataset = subprocess.run(["rm","-rf",kg_notebook_input_data_dir],check=True)
			
			# make dataset dir
			dataset_mkdir = subprocess.run(["mkdir","-p",kg_notebook_input_data_dir],check=True)
			
			# download metadata for an existing dataset
			kg_dataset = subprocess.run(["kaggle","datasets","metadata","-p",kg_notebook_input_data_dir,dataset],check=True,capture_output=True,text=True)
			logger.info(f"Transcript audio dataset metadata log: {kg_dataset}")
			
			# move audio_file to transcript-audio dataset
			cp_audio_file = subprocess.run(["cp",audio_file,kg_notebook_input_data_dir],check=True)
			logger.info(f"transcript audio data: {os.listdir(kg_notebook_input_data_dir)}")
			# create a new dataset version
			kg_dataset_update = subprocess.run(["kaggle","datasets","version","-p",kg_notebook_input_data_dir,"-m","Updated data"])
			
			# check dataset status
			check_dataset_status(dataset)


def update_youtu_url(url,kg_notebook_input_data_dir):
	# prepare new data
	youtube_url_file = 'youtube_url.txt'
	youtube_url_file_path = os.path.join(kg_notebook_input_data_dir,youtube_url_file)

	with open(youtube_url_file_path,'w') as f:
		f.write(url)

def update_kg_youtube_url(dataset,url,kg_notebook_input_data_dir):
	with notebook_update_youtube_url_spinner_placeholder:
		with st.spinner("Processing..."):
			# remove if exists
			rm_dataset = subprocess.run(["rm","-rf",kg_notebook_input_data_dir],check=True)
			
			# make dataset dir
			dataset_mkdir = subprocess.run(["mkdir","-p",kg_notebook_input_data_dir],check=True)
			
			# download metadata for an existing dataset
			kg_dataset = subprocess.run(["kaggle","datasets","metadata","-p",kg_notebook_input_data_dir,dataset],check=True)

			# update youtube url
			update_youtu_url(url,kg_notebook_input_data_dir)

			# create a new dataset version
			kg_dataset_update = subprocess.run(["kaggle","datasets","version","-p",kg_notebook_input_data_dir,"-m","Updated data"])
			
			# check dataset status
			check_dataset_status(dataset)

def update_kg_transcript_model(transcript_model):
	dataset = 'zluckyhou/transcript-model'
	kg_notebook_input_data_dir = 'kg_notebook_input_data_model'
	with notebook_model_initialize_placeholder:
		with st.spinner("Initializing..."):
			# remove if exists
			rm_dataset = subprocess.run(["rm","-rf",kg_notebook_input_data_dir],check=True)
			
			# make dataset dir
			dataset_mkdir = subprocess.run(["mkdir","-p",kg_notebook_input_data_dir],check=True)
			
			# download metadata for an existing dataset
			kg_dataset = subprocess.run(["kaggle","datasets","metadata","-p",kg_notebook_input_data_dir,dataset],check=True)

			# update youtube url
			model_file = 'transcript_model.txt'
			model_file_path = os.path.join(kg_notebook_input_data_dir,model_file)

			with open(model_file_path,'w') as f:
				f.write(transcript_model)

			# create a new dataset version
			kg_dataset_update = subprocess.run(["kaggle","datasets","version","-p",kg_notebook_input_data_dir,"-m","Updated data"])
			
			# check dataset status
			check_dataset_status(dataset)


def progress_function(stream, chunk, bytes_remaining):
	total_size = stream.filesize
	bytes_downloaded = total_size - bytes_remaining
	percentage_of_completion = bytes_downloaded / total_size * 100
	print(f"Downloaded {percentage_of_completion}%")

def youtube_download(video_url, download_path):
	logger.info("Downloading video...")
	try:
		yt = YouTube(video_url, on_progress_callback=progress_function)

		# 获取视频标题
		video_title = yt.title
		audio_length = yt.length
		print(f"Downloading video: {video_title}")

		stream = yt.streams.get_highest_resolution()
		# 获取视频流的默认文件名
		default_filename = stream.default_filename.replace(' ', '_')

		# 下载视频到指定目录
		stream.download(output_path=download_path, filename=default_filename)

		return os.path.join(download_path, default_filename),audio_length
	except Exception as e:
		logger.error(f"youtube download error: {e}")

from moviepy.editor import VideoFileClip

def get_video_duration(file_path):
	try:
		# 加载视频文件
		clip = VideoFileClip(file_path)
		
		# 获取视频时长（以秒为单位）
		duration = clip.duration
		
		# 关闭视频文件
		clip.close()
		
		return duration
	except Exception as e:
		print(f"An error occurred: {e}")
		return None

from pydub import AudioSegment

def get_audio_duration(file_path):
    audio = AudioSegment.from_file(file_path)
    duration_seconds = len(audio) / 1000.0
    return duration_seconds

# import librosa
# def get_audio_duration(file_path):
# 	y, sr = librosa.load(file_path)
# 	# 计算音频时长
# 	duration = librosa.get_duration(y=y, sr=sr)
# 	return duration



def wrap_download_youtube(youtube_url):
	dataset_url = 'zluckyhou/youtube-url'
	kg_notebook_input_data_dir_youtu = 'kg_notebook_input_data_url'
	update_kg_youtube_url(dataset_url,youtube_url,kg_notebook_input_data_dir_youtu)
	# run youtube_download to get youtube video
	notebook_name_youtu = "zluckyhou/youtube-download"
	kg_notebook_dir_youtu = 'kg_notebook_youtu'
	kg_notebook_output_dir_youtu = 'kg_notebook_output_youtu'
	kg_notebook_run_youtube(notebook_name_youtu,kg_notebook_dir_youtu,kg_notebook_output_dir_youtu)
	video_name = [file for file in os.listdir(kg_notebook_output_dir_youtu) if file.endswith('.mp4')][0]
	youtube_video = os.path.join(kg_notebook_output_dir_youtu,video_name)
	
	logger.info(f"youtube video: {youtube_video}")
	st.session_state.youtube_video = youtube_video
	audio_length = get_video_duration(youtube_video)
	st.session_state.audio_length = audio_length





def wrap_transcript_audio(audio_file,target_language):
	st.session_state.translated_srt = ''
	st.session_state.translated_srt_url = ''
	st.session_state.srt_file = ''
	st.session_state.txt_file = ''
	st.session_state.srt_file_url = ''
	st.session_state.txt_file_url = ''
	with transcripting_placeholder:
		with st.spinner("Transcribing..."):
			# remove wav files first
			logger.info(f"audio file for transcript: {audio_file}")
			# rm_wav = subprocess.run(["rm","-rf","part*.wav"],check=True)
			# rm_srt = subprocess.run(["rm","-rf","part*.srt"],check=True)
			# rm_txt = subprocess.run(["rm","-rf","part*.txt"],check=True)
			sorted_split_audio_files = split_audio(audio_file)
			logger.info(f"split files: {sorted_split_audio_files}")
			logger.info("-----------Transcribing------------")
			merged_srt, merged_txt = process_files_concurrently(sorted_split_audio_files,audio_file)

			st.session_state.srt_file = merged_srt
			st.session_state.txt_file = merged_txt
			logger.info(f"srt file: {merged_srt}")
			srt_file_url = upload_file_to_supabase_storage(merged_srt)
			txt_file_url = upload_file_to_supabase_storage(merged_txt)
			st.session_state.srt_file_url = srt_file_url
			st.session_state.txt_file_url = txt_file_url
	if target_language:
		with translate_placeholder:
			with st.spinner("Translating..."):
				translated_srt = wrap_translate(merged_srt,target_language)
				st.session_state.translated_srt = translated_srt
				translated_srt_url = upload_file_to_supabase_storage(translated_srt)
				st.session_state.translated_srt_url = translated_srt_url


def save_uploaded_audio(file_obj):
	base_name = remove_non_ascii(os.path.basename(file_obj.name)).replace(' ', '_')
	mime_type, _ = mimetypes.guess_type(base_name)
	# output_path = 'audio_files_' + st.session_state.user_info.get('name','unknown')
	# remove directory if exists 
	# rm_user_directory = subprocess.run(["rm","-rf",output_path],check=True)
	# mkdir_user_directory = subprocess.run(["mkdir","-p",output_path],check=True)

	output_file_path = base_name
	
	bytes_data = file_obj.getvalue()
	with open(output_file_path,'wb') as f:
		f.write(bytes_data)

	st.session_state.audio_file = output_file_path
	st.session_state.audio_file_type = mime_type

# from st_audiorec import st_audiorec
# def record_and_save_audio():
# 	with st.container(border=True):
# 		wav_audio_data = st_audiorec()

# 		# output_path = 'record_audios'
# 		# remove directory if exists 
# 		# rm_user_directory = subprocess.run(["rm","-rf",output_path],check=True)
# 		# mkdir_user_directory = subprocess.run(["mkdir","-p",output_path],check=True)

# 		# output_file_path = os.path.join(output_path,"record_audio.mp3")

# 		logger.debug(f"record data: {wav_audio_data}")
# 		if wav_audio_data:
# 			with process_record_spinner_placeholder:
# 				with st.spinner("Processing record audio..."):
# 					output_file_path = "record_audio.mp3"
# 					st.session_state.record_audio_data = wav_audio_data
# 					with open(output_file_path,'wb') as f:
# 						f.write(st.session_state.record_audio_data)		
# 					st.session_state.audio_file = output_file_path



def transcript_youtube(youtube_url):
	st.session_state.status = ''
	st.session_state.youtube_video = ''
	st.session_state.srt_file = ''
	st.session_state.quota_limit = ''
	st.session_state.audio_file = ''
	if not youtube_url:
		with empty_url_container:
			st.warning("Please paste a YouTube URL")
		return
	logger.info(f"youtube url:{youtube_url}")
	st.session_state.youtube_url = youtube_url
	if st.session_state.get('user_info', {}):
		with transcript_youtube_spinner_placeholder:
			with st.spinner("Transcription in progress. Sit tight!"):
				user_name = st.session_state.user_info['name']
				email = st.session_state.user_info['email']
				if is_user_valid(email):
					try:
						# update_kg_transcript_model(transcript_model)
						# use youtube-download to download youtube video
						wrap_download_youtube(youtube_url)
						# transcript youtube video
						wrap_transcript_audio(st.session_state.youtube_video,st.session_state.target_language)
						st.session_state.status = 'success'
						update_data = update_user_msg_pv(email)
						# st.session_state.memo = 'success'
					except Exception as e:
						logger.error(f"Transcript running error: {e}")
						st.session_state.status = 'error'
				else:
					st.session_state.quota_limit = "Your free usage has been reached. To continue using the service, please support me by clicking the 'Support Me on Ko-fi' button. Your contribution helps fund further development and unlocks additional usage. Even a small donation makes a big difference - it's like buying me a coffee! Thank you for your support."
					st.session_state.status = 'usage_limit'
					# st.session_state.memo = 'usage limit'
					with free_quota_container:
						st.warning(st.session_state.quota_limit,icon=":material/energy_savings_leaf:")
	else:
		st.session_state.status = 'not_login'
		# st.session_state.memo = 'not login'


def transcript_audio_file(audio_file):
	st.session_state.status = ''
	st.session_state.srt_file = ''
	st.session_state.notebook_status = ''
	st.session_state.quota_limit = ''
	st.session_state.youtube_url = ''
	if not audio_file:
		with empty_file_container:
			st.warning("Please select a file to upload.")
		return
	logger.info(f"audio file:{audio_file}")
	if st.session_state.get('user_info', {}):
		with transcript_audiofile_spinner_placeholder:
			with st.spinner("Transcription in progress. Sit tight!"):
				email = st.session_state.user_info['email']
				if is_user_valid(email):
					try:
						# st.markdown("Transcription task submitted!")
						mime_type, _ = mimetypes.guess_type(audio_file)
						logger.debug(f"mime type: {mime_type}")
						if mime_type.startswith("video"):
							audio_length = get_video_duration(audio_file)
						if mime_type.startswith('audio'):
							audio_length = get_audio_duration(audio_file)
						st.session_state.audio_length = audio_length
						logger.debug(f"audio length: {audio_length}")

						# update_kg_transcript_model(transcript_model)
						# transcript uploaded audio file
						wrap_transcript_audio(audio_file,st.session_state.target_language)
						st.session_state.status = 'success'
						update_data = update_user_msg_pv(email)
						st.session_state.memo = 'success'
					except Exception as e:
						logger.error(f"Transcript running error: {e}")
						st.session_state.status = 'error'
				else:
					st.session_state.quota_limit = "Your free usage has been reached. To continue using the service, please support me by clicking the 'Support Me on Ko-fi' button. Your contribution helps fund further development and unlocks additional usage. Even a small donation makes a big difference - it's like buying me a coffee! Thank you for your support."
					st.session_state.status = 'usage_limit'
					# st.session_state.memo = 'usage limit'
					with free_quota_container:
						st.warning(st.session_state.quota_limit,icon=":material/energy_savings_leaf:")
	else:
		st.session_state.status = 'not_login'
		# st.session_state.memo = 'not login'
		

def update_message():
	msg = {
	"type":st.session_state.trans_type,
	"url":st.session_state.youtube_url,
	"srt":st.session_state.srt_file_url,
	"translated_srt":st.session_state.translated_srt_url,
	"txt":st.session_state.txt_file_url,
	"user_name":st.session_state.user_info.get('name',''),
	"email":st.session_state.user_info.get('email',''),
	"status":st.session_state.status,
	# "memo":st.session_state.memo,
	"audio_file":st.session_state.audio_file,
	"model":"whisper-large-v3"
	}

	supabase_insert_message(table='transcript_messages',message=msg)



if "notebook_status" not in st.session_state:
	st.session_state.notebook_status = 'preparing' 
if "notebook_output" not in st.session_state:
	st.session_state.notebook_output = ''
if "youtube_notebook_status" not in st.session_state:
	st.session_state.youtube_notebook_status = ''
if "youtube_notebook_output" not in st.session_state:
	st.session_state.youtube_notebook_output = ''
if "youtube_video" not in st.session_state:
	st.session_state.youtube_video = ''
if "youtube_url" not in st.session_state:
	st.session_state.youtube_url = ''

if "audio_length" not in st.session_state:
	st.session_state.audio_length = None
if "srt_file_url" not in st.session_state:
	st.session_state.srt_file_url = ''
if "txt_file_url" not in st.session_state:
	st.session_state.txt_file_url = ''
if "srt_file" not in st.session_state:
	st.session_state.srt_file = ''
if "txt_file" not in st.session_state:
	st.session_state.txt_file = ''


if 'user_info' not in st.session_state:
	st.session_state.user_info = {}
if 'status' not in st.session_state:
	st.session_state.status = ''
if 'quota_limit' not in st.session_state:
	st.session_state.quota_limit = ''
if "memo" not in st.session_state:
	st.session_state.memo = ''

if "trans_type" not in st.session_state:
	st.session_state.trans_type = ''

if "audio_file" not in st.session_state:
	st.session_state.audio_file = ''
if "audio_file_type" not in st.session_state:
	st.session_state.audio_file_type = ''
if "record_audio_data" not in st.session_state:
	st.session_state.record_audio_data = ''

if 'target_language' not in st.session_state:
	st.session_state.target_language = ''
if 'translated_srt' not in st.session_state:
	st.session_state.translated_srt = ''
if 'translated_srt_url' not in st.session_state:
	st.session_state.translated_srt_url = ''

# if "model" not in st.session_state:
# 	st.session_state.model = ''

# App title
st.set_page_config(page_title="WhisperFlow",page_icon=":parrot:")


# sidebar
with st.sidebar:
	st.title("🦜 WhisperFlow")
	about = """
	The lightning-fast, AI-powered audio and video transcription solution that will revolutionize your content management workflow.
	"""
	st.markdown(about)

	st.markdown("[![ko-fi](https://wbucijybungpjrszikln.supabase.co/storage/v1/object/public/chatgpt-4o-files/githubbutton_sm_1.svg)](https://ko-fi.com/J3J3YMOKZ)")
	
	from auth0_component import login_button
	
	clientId = st.secrets["auth0_client_id"]
	domain = st.secrets["auth0_domain"]

	
	user_info = st.session_state.get('user_info', {})
	if user_info:
		logger.info(f"User info: {user_info}")
		st.markdown(f"Welcome, {user_info['name']}")
		name = user_info['name']
		user_name = user_info['nickname']
		profile = ''
		picture = user_info['picture']
		oauth_token = user_info['token']
		email = user_info['email']
		# check if user exists
		user_data = supabase_fetch_user_by_email(email)
		if not user_data[1]:
			supabase_insert_user(name,user_name,profile,picture,oauth_token,email)
		if st.button('Logout'):
			st.session_state.user_info = None  # 清除 session 中的用户信息
			st.rerun()  # 重新运行应用以更新状态
	else:
		user_info = login_button(clientId, domain=domain)
		if user_info:
			st.session_state.user_info = user_info  # 保存用户信息到 session
			st.rerun()  # 重新运行应用以更新状态
	st.divider()
	with st.expander("Explore More Apps",icon=":material/apps:"):
		st.link_button(":owl: ChatGPT-4o", "https://chatgpt-4o.streamlit.app/")
		st.link_button(":owl: NativeSpeaker", "https://nativespeaker.streamlit.app/")

	# st.divider()
	# transcript_model = st.selectbox("Transcript model",["medium","large-v2","large-v3"])
	# st.session_state.model = transcript_model
	# st.caption("Medium model is faster, large-v3 offers the best accuracy.")
# st.sidebar.divider()
# st.sidebar.markdown('If you have any questions or need assistance, please feel free to contact me via [email](mailto:hou0922@gmail.com)')



save_kg_json()


st.title("Whisper Flow")
st.markdown(" The lightning-fast, AI-powered audio and video transcription solution that will revolutionize your content management workflow.")


from streamlit_image_select import image_select

img = image_select(
    label="Select Audio Source",
    images=[
        "upload_logo.png",
        "youtube_logo.png",
        # "record_logo.png",
    ],
    captions=["Upload File","YouTube Link"],
)

# col1, col2, col3 = st.columns(3)
# with col1:
# 	youtu_button = st.button("From Youtube url")
# with col2:
# 	record_button = st.button("Record Audio")
# with col3:
# 	upload_button = st.button("Upload File")



if img == 'youtube_logo.png':
# if youtu_button:
	st.session_state.trans_type = 'youtube_url'

	youtube_url = st.text_area("Youtube video url",placeholder="Paste your youtube video url here.").strip()
	
	# need_translate = st.checkbox("Also translate transcription")
	need_translate = st.toggle("Activate Translation")
	if need_translate:
		if is_donation(st.session_state.user_info.get('email')):
			target_language = st.selectbox("Translate into",["简体中文","English","Español","Français","Português","日本語","한국어","Русский"])
			st.session_state.target_language = target_language
		else:
			st.warning('Unlock this feature by supporting me on Ko-fi (link in sidebar)',icon='🔥')
	else:
		st.session_state.target_language = ''

	transcript_youtube_button = st.button(
		label="Transcribe",
		type="primary",
		key="transcript_youtube",
		on_click=transcript_youtube,
		args=[youtube_url],
		)
	if transcript_youtube_button and st.session_state.status:
		logger.debug("update message")
		update_message()

	# transcript_youtube(youtube_url)


elif img == 'upload_logo.png':
# if upload_button:
	st.session_state.trans_type = 'upload_file'
	st.markdown("Need audio? You can easily record it using [vocaroo](https://vocaroo.com/).")
	uploaded_file = st.file_uploader("Upload audio/video", key="audio_file_uploader",type=['mp3','wav','mp4','mpeg','mpga','m4a','webm'])
	if uploaded_file:
		logger.debug(f"upload_file:{uploaded_file}")
		save_uploaded_audio(uploaded_file)
	if st.session_state.audio_file_type.startswith('audio'):
		st.audio(st.session_state.audio_file,format=st.session_state.audio_file_type)
	if st.session_state.audio_file_type.startswith('video'):
		st.video(st.session_state.audio_file,format=st.session_state.audio_file_type)
	
	# need_translate = st.checkbox("Also translate transcription")
	need_translate = st.toggle("Activate Translation")
	if need_translate:
		if is_donation(st.session_state.user_info.get('email')):
			target_language = st.selectbox("Translate into",["简体中文","English","Español","Français","Português","日本語","한국어","Русский"])
			st.session_state.target_language = target_language
		else:
			st.warning('Unlock this feature by supporting me on Ko-fi (link in sidebar)',icon='🔥')
	else:
		st.session_state.target_language = ''

	transcript_audio_button = st.button(
		label="Transcribe",
		type="primary",
		key="transcript_audio",
		on_click=transcript_audio_file,
		args=[st.session_state.audio_file],
		# disabled = not st.session_state.audio_file
		)
	if transcript_audio_button and st.session_state.status:
		logger.debug("update message")
		update_message()

	# transcript_audio_file(st.session_state.audio_file)


# elif img == 'record_logo.png':
# # if record_button:
# 	st.session_state.trans_type = 'record_audio'
# 	process_record_spinner_placeholder = st.empty()
# 	record_and_save_audio()
# 	if st.session_state.record_audio_data:
# 		logger.info(f"record audio: {st.session_state.record_audio_data}")
# 		st.audio(st.session_state.record_audio_data, format='audio/wav')
# 	transcript_record_button = st.button(
# 		label="Transcript",
# 		type="primary",
# 		key="transcript_record",
# 		on_click=transcript_audio_file,
# 		args=[st.session_state.audio_file],
# 		disabled = not st.session_state.audio_file
# 		)
# 	st.markdown("---")
# 	transcript_audiofile_spinner_placeholder = st.empty()
# 	login_tip_container = st.container()
# 	if transcript_record_button and st.session_state.status:
# 		update_message()

st.markdown("---")

if not st.session_state.user_info:
	st.warning("Please click the 'Login' button in the sidebar to proceed.", icon=":material/passkey:")


youtube_video_placeholder = st.empty()

if st.session_state.trans_type == 'youtube_url' and st.session_state.youtube_video:
	with youtube_video_placeholder:
		st.video(st.session_state.youtube_video)

empty_file_container = st.container()
empty_url_container = st.container()
free_quota_container = st.container()
transcript_youtube_spinner_placeholder = st.empty()
transcript_audiofile_spinner_placeholder = st.empty()

notebook_model_initialize_placeholder = st.empty()
notebook_update_youtube_url_spinner_placeholder = st.empty()
notebook_data_spinner_placeholder = st.empty()
notebook_pull_spinner_placeholder = st.empty()
notebook_running_spinner_placeholder = st.empty()
transcripting_placeholder = st.empty()
translate_placeholder = st.empty()
notebook_save_output_spinner_placeholder = st.empty()


if st.session_state.status == 'success':
	
	subtitle = st.session_state.translated_srt if st.session_state.translated_srt else st.session_state.srt_file

	if st.session_state.trans_type == 'youtube_url' and st.session_state.youtube_video:
		with youtube_video_placeholder:
			st.video(st.session_state.youtube_video,subtitles=subtitle)
	# st.markdown(f"Transcription completed! Download [Audio subtitle]({st.session_state.srt_file_url}) or [Transcription in plain text]({st.session_state.txt_file_url})")
	st.markdown("Transcription completed! If you need to organize or summarize the text, try [ChatGPT-4o](https://chatgpt-4o.streamlit.app/)")
	# col1,col2 = st.columns(2)

	with open(subtitle) as file:
		st.download_button(
			label="Download subtitle srt file",
			data=file,
			file_name="audio_subtitle.srt",
			mime=mimetypes.guess_type(st.session_state.srt_file)[0]
			)

	# with open(st.session_state.txt_file) as file:
	# 	st.download_button(
	# 		label="Download transcription in plain text",
	# 		data=file,
	# 		file_name="audio_transcription.txt",
	# 		mime=mimetypes.guess_type(st.session_state.txt_file)[0]
	# 		)

	# st.markdown("---")

	st.markdown("Preview")
	with st.container(border=True):
		with open(subtitle) as f:
			subtitle_txt = f.read()
		st.markdown(f"{subtitle_txt[:1000]}")
	# st.text_area(label='Transcription Preview',value=f"{plain_transcript[:1000]}",height=500)


if st.session_state.status == 'error':
	st.error("Opps,something went wrong!",icon="🔥")


