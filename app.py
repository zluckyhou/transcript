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
	base_name = remove_non_ascii(os.path.basename(file_path))
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
		return True
	elif donation_data[1]:
		return True
	elif msg_pv < 3:
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
					logger.info("The notebook is still running. Checking again in 5 seconds...")
					time.sleep(interval)  # 等待5秒

def check_kernel_status_transcript(notebook, interval=5):
	# 预估总时长
	estimate_time = 237 + st.session_state.video_length / 25
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
					logger.info("The notebook is still running. Checking again in 5 seconds...")
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
		video_length = yt.length
		print(f"Downloading video: {video_title}")

		stream = yt.streams.get_highest_resolution()
		# 获取视频流的默认文件名
		default_filename = stream.default_filename.replace(' ', '_')

		# 下载视频到指定目录
		stream.download(output_path=download_path, filename=default_filename)

		return os.path.join(download_path, default_filename),video_length
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
	video_length = get_video_duration(youtube_video)
	logger.info(f"youtube video: {youtube_video}")

	st.session_state.youtube_video = youtube_video
	st.session_state.video_length = video_length

def wrap_transcript_audio(audio_file):
	# use audio-transcript-forapi to transcript
	dataset = 'zluckyhou/transcript-audio'
	# youtube_url = "https://www.youtube.com/watch?v=JUSELxessnU&ab_channel=WIRED"
	kg_notebook_input_data_dir = 'kg_notebook_input_data'		

	# move youtube video to transcript-audio dataset
	update_transcript_audio(dataset,audio_file,kg_notebook_input_data_dir)

	# st.markdown(f"Notebook data: {notebook_data}")

	notebook_name = "zluckyhou/audio-transcript-forapi"
	kg_notebook_dir = 'kg_notebook/'
	kg_notebook_output_dir = 'kg_notebook_output'


	# run kaggle
	kg_notebook_run_with_transcript(notebook_name,kg_notebook_dir,kg_notebook_output_dir)

	# display result if notebook running complete
	if st.session_state.notebook_output:
		output_files = os.listdir(kg_notebook_output_dir)
		srt_file = [file for file in output_files if file.endswith('.srt')][0]
		txt_file = [file for file in output_files if file.endswith('.txt')][0]
		st.session_state.srt_file = os.path.join(kg_notebook_output_dir,srt_file)
		logger.info(f"srt file: {srt_file}")
		srt_file_url = upload_file_to_supabase_storage(os.path.join(kg_notebook_output_dir,srt_file))
		txt_file_url = upload_file_to_supabase_storage(os.path.join(kg_notebook_output_dir,txt_file))
		st.session_state.srt_file_url = srt_file_url
		st.session_state.txt_file_url = txt_file_url
	else:
		st.error("Opps,something went wrong!",icon="🔥")




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
if "audio_file" not in st.session_state:
	st.session_state.audio_file = ''
if "video_length" not in st.session_state:
	st.session_state.video_length = None
if "srt_file_url" not in st.session_state:
	st.session_state.srt_file_url = ''
if "txt_file_url" not in st.session_state:
	st.session_state.txt_file_url = ''
if "srt_file" not in st.session_state:
	st.session_state.srt_file = ''


if 'user_info' not in st.session_state:
	st.session_state.user_info = {}
if 'status' not in st.session_state:
	st.session_state.status = ''
if 'quota_limit' not in st.session_state:
	st.session_state.quota_limit = ''



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
	

st.sidebar.divider()
st.sidebar.markdown('If you have any questions or need assistance, please feel free to contact me via [email](mailto:hou0922@gmail.com)')





st.title("Whisper Flow")
st.markdown(" The lightning-fast, AI-powered audio and video transcription solution that will revolutionize your content management workflow.")


save_kg_json()

youtube_url = st.text_area("Youtube video url",placeholder="Paste your youtube video url here.")

transcript_button = st.button(label="Transcript",type="primary")

if transcript_button:

	st.session_state.youtube_video = ''
	st.session_state.srt_file = ''
	
	st.markdown("---")

	if not youtube_url:
		st.warning("Please input your youtube video url",icon=":material/warning:")
	elif youtube_url:
		logger.info(f"youtube url:{youtube_url}")
		
		notebook_update_youtube_url_spinner_placeholder = st.empty()
		notebook_data_spinner_placeholder = st.empty()
		notebook_pull_spinner_placeholder = st.empty()
		notebook_running_spinner_placeholder = st.empty()
		notebook_save_output_spinner_placeholder = st.empty()

		trans_type = 'youtube_url'
		if st.session_state.get('user_info', {}):
			user_name = st.session_state.user_info['name']
			email = st.session_state.user_info['email']
			if is_user_valid(email):
				# use youtube-download to download youtube video
				wrap_download_youtube(youtube_url)
				# transcript youtube video
				if st.session_state.youtube_video:
					wrap_transcript_audio(st.session_state.youtube_video)
				st.session_state.status = 'success'
				update_data = update_user_msg_pv(email)

			else:
				st.session_state.quota_limit = True
				st.session_state.status = 'failed'
				memo = "Your free usage has been reached. To continue using the service, please support me by clicking the 'Support Me on Ko-fi' button. Your contribution helps fund further development and unlocks additional usage. Even a small donation makes a big difference - it's like buying me a coffee! Thank you for your support."
				st.warning(memo,icon=":material/energy_savings_leaf:")
		else:
			memo = "Please click the 'Login' button in the sidebar to proceed."
			st.session_state.status = 'failed'
			st.warning(memo, icon=":material/passkey:")
	msg = {
	"type":trans_type,
	"url":youtube_url,
	"srt":st.session_state.srt_file_url,
	"txt":st.session_state.txt_file_url,
	"user_name":st.session_state.user_info.get('name',''),
	"email":st.session_state.user_info.get('email',''),
	"status":st.session_state.status,
	"memo":memo
	}

	supabase_insert_message(table='transcript_messages',message=msg)



video_placeholder = st.empty()


if st.session_state.youtube_video:
	with video_placeholder:
		st.video(st.session_state.youtube_video)
if st.session_state.srt_file:
	with video_placeholder:
		st.video(st.session_state.youtube_video,subtitles=st.session_state.srt_file)
		st.markdown("Transcription completed successfully!")
		st.markdown(f"Download [video subtitle]({srt_file_url}) or [Transcript in plain text]({txt_file_url})")


