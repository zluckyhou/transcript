import os
import subprocess
from groq import Groq
import json
import re
import concurrent.futures
import time
import streamlit as st


# 自定义排序键函数，提取文件名中的编号
def get_file_number(filename):
    match = re.search(r'part_(\d+).wav', filename)
    return int(match.group(1)) if match else float('inf')

def split_audio(audio_file):
    # 第一步：将音频文件降采样并转换为单声道
    reduced_audio_file = 'reduced_audio.wav'
    rm_reduced_audio = subprocess.run(["rm","-rf",reduced_audio_file],check=True)
    rm_wav = subprocess.run(["rm","-rf","part*.wav"],check=True)
    reduce_command = [
        'ffmpeg',
        '-y',
        '-i', audio_file,
        '-ar', '16000',
        '-ac', '1',
        '-map', '0:a',
        reduced_audio_file
    ]

    # 运行降采样命令
    subprocess.run(reduce_command, check=True)

    # 第二步：将降采样后的音频文件拆分为多个部分
    segment_command = [
        'ffmpeg',
        '-y',
        '-i', reduced_audio_file,
        '-f', 'segment',
        '-segment_time', '300',
        '-c', 'copy',
        'part_%d.wav'
    ]

    # 运行拆分命令
    subprocess.run(segment_command, check=True)


    # 第三步：列出所有拆分后的文件
    split_audio_files = [f for f in os.listdir('.') if f.startswith('part_') and f.endswith('.wav')]

    # 按编号排序文件列表
    sorted_split_audio_files = sorted(split_audio_files, key=get_file_number)

    return sorted_split_audio_files


def seconds_to_hms(seconds):
    # Simple conversion of seconds to HH:MM:SS format
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

# 转为txt文件
def segments_to_txt(segments,output_file,segment_idx):
    with open(output_file,'w',encoding='utf8') as f:
        for segment in segments:
            # Converting start time to HH:MM:SS format
            start_time = seconds_to_hms(segment.get('start')+300*segment_idx)
            text = segment.get('text').strip()  # Removing any leading/trailing whitespaces from the text
            f.write(f"{start_time}: {text}\n\n")
    print(
        f"Voila!✨ Transcript plain text file saved 👉 {output_file}"
    )

# Function to convert time in seconds to SRT time format
def convert_to_srt_time(timestamp):
    hours = int(timestamp // 3600)
    minutes = int((timestamp % 3600) // 60)
    seconds = int(timestamp % 60)
    milliseconds = int((timestamp % 1) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

# 转为srt文件
def segments_to_srt(segments,output_file,segment_idx):
    with open(output_file,'w',encoding='utf8') as f:
        for idx,segment in enumerate(segments):
            # Converting start time to HH:MM:SS format
            start_time = convert_to_srt_time(segment.get('start')+300*segment_idx)
            end_time = convert_to_srt_time(segment.get('end')+300*segment_idx)
            text = segment.get('text').strip()  # Removing any leading/trailing whitespaces from the text
            f.write(f"{idx + 1}\n{start_time} --> {end_time}\n{text}\n\n")
    print(
        f"Voila!✨ Srt file saved 👉 {output_file}"
    )


def transcript(filename):

    client = Groq(api_key=st.secrets['GROQ_API_KEY'])
    segment_idx = int(get_file_number(filename))

    filename = os.path.join(os.getcwd(),filename)
    
    with open(filename, "rb") as file:
        transcription = client.audio.transcriptions.create(
          file=(filename, file.read()),
          model="whisper-large-v3",
    #      prompt="Specify context or spelling",  # Optional
          response_format="verbose_json",  # Optional
          # language="en",  # Optional
          temperature=0.0  # Optional
        )
    segments = transcription.segments

    srt_output_file = filename.replace('.wav','.srt')

    txt_output_file = filename.replace('.wav','.txt')

    segments_to_srt(segments,srt_output_file,segment_idx)

    segments_to_txt(segments,txt_output_file,segment_idx)
    
    return srt_output_file,txt_output_file


def process_files_concurrently(file_list,merged_filename):
    max_workers = 20  # 最大并行任务数
    delay_between_requests = 4  # 每分钟最多20个请求，计算得每3秒一个请求
    
    file_list = [os.path.join(os.getcwd(),file) for file in file_list]
    # 定义存储结果的字典
    results = {}

    # 使用ThreadPoolExecutor进行并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_filename = {executor.submit(transcript, filename): filename for filename in file_list}
        
        for future in concurrent.futures.as_completed(future_to_filename):
            filename = future_to_filename[future]
            try:
                srt_file, txt_file = future.result()
                results[filename] = (srt_file, txt_file)
            except Exception as exc:
                print(f'{filename} generated an exception: {exc}')
            time.sleep(delay_between_requests)

    # 根据输入文件列表顺序获取结果
    srt_files = [results[filename][0] for filename in file_list]
    txt_files = [results[filename][1] for filename in file_list]
    
    print(f'check str files order: {srt_files}')
    print(f'check txt files order: {txt_files}')
    
    # 合并SRT和TXT文件
    merged_srt_file = os.path.splitext(merged_filename)[0] + '.srt'
    merged_txt_file = os.path.splitext(merged_filename)[0] + '.txt'

    with open(merged_srt_file, 'w') as srt_out:
        for srt_file in srt_files:
            with open(srt_file, 'r') as srt_in:
                srt_out.write(srt_in.read() + '\n')

    with open(merged_txt_file, 'w') as txt_out:
        for txt_file in txt_files:
            with open(txt_file, 'r') as txt_in:
                txt_out.write(txt_in.read() + '\n')

    return merged_srt_file, merged_txt_file



# sorted_split_audio_files = split_audio(audio_file)

# def wrap_transcript_audio(audio_file):
#   sorted_split_audio_files = split_audio(audio_file)
#   merged_srt, merged_txt = process_files_concurrently(sorted_split_audio_files)
#   return merged_srt,merged_txt
