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



# Configure logger
logging.basicConfig(format="\n%(asctime)s\n%(message)s", level=logging.INFO, force=True)



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
		with st.spinner("Initializing WhisperFlow's advanced AI transcription engine."):
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
		while True:
			result = subprocess.run(["kaggle", "kernels", "status", notebook], capture_output=True, text=True)
			stdout = result.stdout
			# 提取状态值
			status = re.findall(r'status "(\w+)"',stdout)[0]
			st.session_state.youtube_notebook_status = status
			logging.info(f"The current status is: {status}")
			# 检查状态是否为"complete"
			if status == "complete":
				logging.info("The notebook has finished running.")
				break
			elif status == 'error':
				logging.info("Something wrong, please check the notebook status.")
				break
			else:
				logging.info("The notebook is still running. Checking again in 5 seconds...")
				time.sleep(interval)  # 等待5秒

def check_kernel_status_transcript(notebook, interval=5):
	# 预估总时长
	estimate_time = 237 + st.session_state.video_length / 25
	start_time = time.time()
	with notebook_running_spinner_placeholder:
		# 初始化进度条
		progress_text = "WhisperFlow is now actively transcribing your audio/video."
		my_bar = st.progress(0, text=progress_text)
		# 无限循环，直到状态变为"complete"
		while True:
			result = subprocess.run(["kaggle", "kernels", "status", notebook], capture_output=True, text=True)
			stdout = result.stdout
			# 提取状态值
			status = re.findall(r'status "(\w+)"',stdout)[0]
			st.session_state.notebook_status = status
			logging.info(f"The current status is: {status}")

			elapsed_time = time.time() - start_time  # 计算已经过去的时间
			progress = min(elapsed_time / estimate_time, 0.99)  # 计算进度百分比，最大为0.99
			my_bar.progress(int(progress * 100), text=progress_text)
			
			# 检查状态是否为"complete"
			if status == "complete":
				logging.info("The notebook has finished running.")
				my_bar.progress(100, text="Transcription completed！")
				break
			elif status == 'error':
				logging.info("Something wrong, please check the notebook status.")
				break
			else:
				logging.info("The notebook is still running. Checking again in 5 seconds...")
				time.sleep(interval)  # 等待5秒


def save_output(notebook,kg_notebook_output_dir):
	with notebook_save_output_spinner_placeholder:
		with st.spinner("Your transcription is almost ready! "):
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
		logging.info('output file saved success!')
	else:
		logging.error("opps! something wrong")

def kg_notebook_run_with_transcript(notebook,kg_notebook_dir,kg_notebook_output_dir):	
	set_notebook_dir(kg_notebook_dir)
	
	pull_and_run_notebook(notebook,kg_notebook_dir)
	check_kernel_status_transcript(notebook, interval=5)
	if st.session_state.notebook_status == "complete":
		save_output(notebook,kg_notebook_output_dir)
		st.session_state.notebook_output = True
		logging.info('output file saved success!')
	else:
		logging.error("opps! something wrong")




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
			logging.info(f"doload dataset metadata log: {kg_dataset}")
			
			# move audio_file to transcript-audio dataset
			cp_audio_file = subprocess.run(["cp",audio_file,kg_notebook_input_data_dir],check=True)

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
	logging.info("Downloading video...")
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
		logging.error(f"youtube download error: {e}")

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
if "video_length" not in st.session_state:
	st.session_state.video_length = None


# App title
st.set_page_config(page_title="WhisperFlow",page_icon=":parrot:")


st.title("Whisper FLow")
st.markdown(" The lightning-fast, AI-powered audio and video transcription solution that will revolutionize your content management workflow.")


save_kg_json()

youtube_url = st.text_area("Youtube video url",placeholder="Paste your youtube video url here.")

transcript_button = st.button(label="Transcript",type="primary")

if transcript_button:
	if not youtube_url:
		st.warning("Please input your youtube video url",icon=":material/warning:")
	elif youtube_url:
		logging.info(f"youtube url:{youtube_url}")

		######### use youtube-download to download youtube video
		# dataset = 'zluckyhou/transcript-audio'
		dataset_url = 'zluckyhou/youtube-url'
		kg_notebook_input_data_dir_youtu = './kg_notebook_input_data_url'
		update_kg_youtube_url(dataset_url,youtube_url,kg_notebook_input_data_dir_youtu)
		# run youtube_download to get youtube video
		notebook_name_youtu = "zluckyhou/youtube-download"
		kg_notebook_dir_youtu = './kg_notebook_youtu'
		kg_notebook_output_dir_youtu = './kg_notebook_output_youtu'
		kg_notebook_run_youtube(notebook_name_youtu,kg_notebook_dir_youtu,kg_notebook_output_dir_youtu)
		youtube_video = os.path.join(kg_notebook_output_dir_youtu,os.listdir(kg_notebook_output_dir_youtu)[0])
		video_length = get_video_duration(youtube_video)

		st.session_state.youtube_video = youtube_video
		st.session_state.video_length = video_length

		# use audio-transcript-forapi to transcript
		dataset = 'zluckyhou/transcript-audio'
		# youtube_url = "https://www.youtube.com/watch?v=JUSELxessnU&ab_channel=WIRED"
		kg_notebook_input_data_dir = './kg_notebook_input_data'		
		notebook_data_spinner_placeholder = st.empty()
		# move youtube video to transcript-audio dataset
		update_transcript_audio(dataset,youtube_video,kg_notebook_input_data_dir)

		st.markdown("---")
		# st.markdown(f"Notebook data: {notebook_data}")

		notebook_name = "zluckyhou/audio-transcript-forapi"
		kg_notebook_dir = './kg_notebook/'
		kg_notebook_output_dir = './kg_notebook_output'

		notebook_pull_spinner_placeholder = st.empty()
		notebook_running_spinner_placeholder = st.empty()
		notebook_save_output_spinner_placeholder = st.empty()
		# run kaggle
		kg_notebook_run_with_transcript(notebook_name,kg_notebook_dir,kg_notebook_output_dir)

		# display result if notebook running complete
		if st.session_state.notebook_output:
			st.markdown("Transcription completed successfully!")
			output_files = os.listdir(kg_notebook_output_dir)
			videos = [file for file in output_files if mimetypes.guess_type(file)[0].startswith('video')]
			video_file = videos[0] if videos else ''
			srt_file = [file for file in output_files if file.endswith('.srt')]
			txt_file = [file for file in output_files if file.endswith('.txt')]
			st.video(video_file,subtitles=srt_file)
			st.markdown(f"Download [video subtitle](srt_file) or [Transcript in plain text](txt_file)")
		else:
			st.error("Opps,something went wrong!",icon="🔥")

