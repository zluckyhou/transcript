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
	# pull notebook code and metadata
	pull = subprocess.run(["kaggle","kernels","pull",notebook,"-p",kg_notebook_dir,"-m"],check=True)
	# check notebook metadata	 
	with open(os.path.join(kg_notebook_dir,'kernel-metadata.json')) as f:
		kg_metadata = json.load(f)
	
	# push notebook
	kg_push = subprocess.run(["kaggle", "kernels", "push", "-p", kg_notebook_dir], check=True)
	return kg_metadata


def check_kernel_status(notebook, interval=5):
	with notebook_running_spinner_placeholder:
		with st.spinner(f"Transcript job status: {st.session_state.notebook_status}"):
			# 无限循环，直到状态变为"complete"
			while True:
				result = subprocess.run(["kaggle", "kernels", "status", notebook], capture_output=True, text=True)
				stdout = result.stdout
				# 提取状态值
				status = re.findall(r'status "(\w+)"',stdout)[0]
				st.session_state.notebook_status = status
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

def save_output(notebook,kg_notebook_output_dir):
	
	# remove kg_notebook dir if exists
	kg_rm_output = subprocess.run(["rm","-rf",kg_notebook_output_dir],check=True)
	# create output dir
	kg_mkdir_output = subprocess.run(["mkdir", "-p", kg_notebook_output_dir], check=True)
	# save output
	kg_save_output = subprocess.run(["kaggle", "kernels", "output", notebook, "-p", kg_notebook_output_dir], check=True)

def kg_notebook_run(notebook,kg_notebook_dir,kg_notebook_output_dir):	
	set_notebook_dir(kg_notebook_dir)
	
	pull_and_run_notebook(notebook,kg_notebook_dir)
	status = check_kernel_status(notebook, interval=5)
	if status == "complete":
		save_output(notebook,kg_notebook_output_dir)
		print('output file saved success!')
	else:
		print("opps! something wrong")
	
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

def update_data(dataset,url,kg_notebook_input_data_dir):
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



def update_youtu_url(url,kg_notebook_input_data_dir):
	# prepare new data
	youtube_url_file = 'youtube_url.txt'
	youtube_url_file_path = os.path.join(kg_notebook_input_data_dir,youtube_url_file)

	with open(youtube_url_file_path,'w') as f:
		f.write(url)

if "notebook_status" not in st.session_state:
	st.session_state.notebook_status = 'preparing' 


# App title
st.set_page_config(page_title="WhisperFlow",page_icon=":parrot:")


st.title("Whisper FLow")
st.markdown(" The lightning-fast, AI-powered audio and video transcription solution that will revolutionize your content management workflow.")


save_kg_json()

youtube_url = st.text_input("Youtube video url",placeholder="Paste your youtube video url here.")

transcript_button = st.button(label="Transcript",type="primary")

if transcript_button:
	if not youtube_url:
		st.warning("Please input your youtube video url",icon=":material/warning:")
	elif youtube_url:
		dataset = 'zluckyhou/transcript-audio'
		# youtube_url = "https://www.youtube.com/watch?v=JUSELxessnU&ab_channel=WIRED"
		kg_notebook_input_data_dir = './kg_notebook_input_data'

		update_data(dataset,youtube_url,kg_notebook_input_data_dir)

		notebook_data = os.listdir(kg_notebook_input_data_dir)

		st.markdown("---")
		# st.markdown(f"Notebook data: {notebook_data}")

		notebook_name = "zluckyhou/audio-transcript-forapi"
		kg_notebook_dir = './kg_notebook/'
		kg_notebook_output_dir = './kg_notebook_output'

		notebook_running_spinner_placeholder = st.empty()
		# run kaggle
		kg_notebook_run(notebook,kg_notebook_dir,kg_notebook_output_dir)

		# display result if notebook running complete
		if st.session_state.notebook_status == 'complete':
			output_files = os.listdir(kg_notebook_output_dir)
			videos = [file for file in output_files if mimetypes.guess_type(file)[0].startswith('video')]
			video_file = videos[0] if videos else ''
			srt_file = [file for file in output_files if file.endswith('.srt')]
			txt_file = [file for file in output_files if file.endswith('.txt')]
			st.video(video_file,subtitles=srt_file)
			st.markdown(f"Download [video subtitle](srt_file) or [Transcript in plain text](txt_file)")
		if st.session_state.notebook_status == 'error':
			st.error("Opps,something went wrong!",icon="🔥")
	



