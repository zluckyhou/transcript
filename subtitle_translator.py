from openai import OpenAI
import tiktoken
import streamlit as st
import os

# string tokens
def num_tokens_from_string(string: str, model: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(string))
    return num_tokens
# Define response function
def get_completion(prompt,system_prompt="You are a helpful assistant.",model="gpt-3.5-turbo"):
    openai_api_key = st.secrets['openai_api_key']
    base_url = st.secrets['burn_base_url']

    client = OpenAI(base_url=base_url, api_key=openai_api_key)

    temperature = 0.7
    max_tokens = 4000

    messages=[
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": prompt}
    ]
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    #       stream=True,
    #       stream_options={"include_usage":True}
    )
    return completion.choices[0].message.content

system_prompt = """你是一个专业的翻译家，擅长多语言翻译。你的目标是将提供给你的字幕文本，翻译为{language}。
提供给你的文本可能包含多句话，两句话之间以`\n\n`进行分隔，每句话包含了编号、时间戳、内容。

在翻译时，请严格请按照以下步骤, think step by step：
step1: 根据`\n\n`分隔符将内容进行拆分，每个句子为一个单位进行翻译；
step2: 针对每个句子：
 - 识别这句话的编号、时间戳、原始内容；
 - 保留编号、时间戳、原始内容;
 - 将原始内容翻译为{language}，翻译时请不要僵硬地一句句翻译，而是在理解原文的基础上进行意译，确保翻译的结果流畅、自然，符合{language}的使用习惯；[This is VERY IMPORTANT]
 - 将翻译后的内容，附加到这个句子最后。

### 下面是一个例子，原始内容为英文，需要翻译为简体中文

Input:

62
00:02:53,359 --> 00:02:54,419
I did nothing wrong.


Output:

62
00:02:53,359 --> 00:02:54,419
I did nothing wrong.
我没有做错任何事。


"""


def split_text_by_token_length(text: str, delimiter: str, chunk_token: int, model: str) -> list:
    """Splits the text by delimiter and ensures each part is within the chunk_token limit."""
    parts = text.split(delimiter)
    result = []
    current_chunk = []
    current_length = 0
    
    for part in parts:
        part_length = num_tokens_from_string(part, model)
        if current_length + part_length > chunk_token:
            # If adding this part exceeds the limit, finalize the current chunk
            if current_chunk:
                result.append(delimiter.join(current_chunk))
            current_chunk = [part]
            current_length = part_length
        else:
            # Otherwise, add the part to the current chunk
            current_chunk.append(part)
            current_length += part_length
            
    # Don't forget to add the last chunk
    if current_chunk:
        result.append(delimiter.join(current_chunk))
    
    return result




import concurrent.futures

# 高并发处理翻译
def process_subtitle_chunks(subtitle_chunks, system_prompt, model):
    results = [None] * len(subtitle_chunks)  # 预先分配一个与 subtitle_chunks 等长的列表
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(get_completion, chunk, system_prompt, model): idx
            for idx, chunk in enumerate(subtitle_chunks)
        }
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]  # 获取任务的索引
            try:
                result = future.result()
                results[idx] = result  # 将结果放在预先分配的列表的正确位置
                print(f"Completed task {idx+1}/{len(subtitle_chunks)}")  # 打印当前任务的进度
            except Exception as e:
                print(f"Error in task {idx+1}: {e}")
    return results


def wrap_translate(srt_file,language,system_prompt=system_prompt):

    with open(srt_file) as f:
        subtitle_en = f.read()

    # split text
    subtitle_en_splits = split_text_by_token_length(subtitle_en.strip(),delimiter='\n\n',chunk_token=1000,model=st.secrets['chat_model'])
    subtitle_multi_splits = process_subtitle_chunks(subtitle_en_splits, system_prompt.format(language=language), model=st.secrets['chat_model'])


    multilingo_subtitle = '\n\n'.join(subtitle_multi_splits)

    multilingo_filename = os.path.splitext(srt_file)[0] + '_multilingo' + os.path.splitext(srt_file)[1]
    with open(multilingo_filename,'w',encoding='utf-8') as f:
        f.write(multilingo_subtitle)

    return multilingo_filename
