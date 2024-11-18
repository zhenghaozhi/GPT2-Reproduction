
# 拼接prompt头
heads = r'我现在需要你对json文段中的"text"字段做如下处理：对此文段的关键信息提出3至6个不同的关键问题并给予这些问题相应的回答(一定要源自于原文段而不是你的理解，问题要在原文中有答案），问题不能相同，问题个数根据文段承载信息多少决定。每个问题应当能够独立于文段而被理解，所有问题的设问对象一定是一个明确的主体而非代词或模糊描述。生成的内容格式要求的正则表达式为: {"question": "(.+?)", "answer": "(.+?)"} ,以这样的json形式输出给我，需要严格遵守此格式。不管你生成几对{"question": "", "answer": ""}，问题和答案都应该是一一对应的，一个返回的json中包含一个问题和一个答案。对于我给出的文段，直接返回给我3到6个按要求的形式的json封装。另外要求所有问题的主谓宾明确，不要出现代词或指代词。问题均不超过15字不少于1字，答案均不超过300字不少于1字。下面是文段：'

from openai import OpenAI
import os
import json
import time
import re

# 本地挂载qwen2的端口
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

directory = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(directory, 'sft_saves.jsonl')

pattern = re.compile(r'{"question": "(.+?)", "answer": "(.+?)"}')
error_log_path = os.path.join(directory, 'sft_error_log.txt')
erro_log = open(error_log_path, 'a', encoding='utf-8')


def generation(inputs, lineid, failed_times):
    system_content = "你是一个智能助手，你总是提供合理的、正确的、有用的回答。"
    if (failed_times >= 3): # 生成错误后提示
        system_content += "刚才你的回答错误。注意问题中不要出现代词，问题中也不要出现未明确的或可具体说明是什么的事物。你只需要生成两条问题就可以了。"
    else: # 确保每个词条间不会相互影响
        system_content += "这是一次全新的对话。"
    if (failed_times >= 6): # 错误时提示并记录错误行数
        erro_log.write("ERROR: " + str(lineid) + "\n")
        output_data = []
        return output_data
    completion = client.chat.completions.create(
        model="Qwen/Qwen2-7B-Instruct-GGUF",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "assistant", "content": inputs}
        ],
        temperature=0.45,
    ) # 向api发送请求
    outputs = str(completion.choices[0].message.content)
    matches = pattern.findall(outputs)
    output_data = [{"question": match[0], "answer": match[1], "id": lineid} for match in matches] # 尝试匹配json形式输出
    
    if (len(output_data) <= 1 or len(output_data) > 6): # 若数量不符合要求则重新生成
        print("ERROR on generating, retrying...")
        print("ERROR output: " + outputs)
        # time.sleep(1)
        return generation(inputs, lineid, failed_times + 1)

    str1 = '(.+?)'
    str2 = '(+?)'
    str3 = '问题'
    str4 = '答案'
    str5 = r'\((.*?)\)'
    str6 = ["这", "此", "该", "他", "她", "它"] # 本地大模型生成内容格式检测
    
    pattern_1 = re.compile(re.escape(str1))
    pattern_2 = re.compile(re.escape(str2))
    pattern_3 = re.compile(str3)
    pattern_4 = re.compile(str4)
    pattern_5 = re.compile(str5)
    pattern_6 = re.compile('|'.join(map(re.escape, str6)))
    matches_1 = True
    matches_2 = True
    matches_34 = True
    matches_5 = True
    matches_6 = True
    
    for odata in output_data:
        matches_1 = matches_1 and not (pattern_1.search(odata["question"]) or pattern_1.search(odata["answer"]))
        matches_2 = matches_2 and not (pattern_2.search(odata["question"]) or pattern_2.search(odata["answer"]))
        matches_34 = matches_34 and not (pattern_3.search(odata["question"]) and pattern_4.search(odata["answer"]))
        matches_5 = matches_5 and not (pattern_5.fullmatch(odata["question"]) or pattern_5.fullmatch(odata["answer"]))
        matches_6 = matches_6 and not pattern_6.search(odata["question"])
        
    if matches_1 and matches_2 and matches_34 and matches_5 and matches_6: # 若出现常见异常现象则重新生成
        return output_data
    else:
        print("ERROR on matching, retrying...")
        print("ERROR output: " + outputs)
        # time.sleep(1)
        return generation(inputs, lineid, failed_times + 1)

current_id = 0

with open(os.path.join(directory, "sft_saves.jsonl"), 'r', encoding='utf-8') as f: # 断点继续功能
    for line_cnt, line in enumerate(f, start=1):
        json_read = json.loads(line)
        if int(json_read["id"]) != current_id and int(json_read["id"]) != current_id + 1:
            print("ERROR on id reading: " + str(json_read["id"]))
            erro_log.write("MISSING: " + str(json_read["id"]) + "\n")
        current_id = int(json_read["id"])


with open(os.path.join(directory, "sft_prepare_data.jsonl"), 'r', encoding='utf-8') as f:
    for line_cnt, line in enumerate(f, start=1): # 对于每一条wiki生成对应的QA对
        if line_cnt <= current_id: # 断点继续
            continue
        
        # if line_cnt == 30:
        #     break
        input_data = heads + line # 拼接prompt
        
        output_data = generation(input_data, line_cnt, 0) # 向api申请得到合法json
        if (len(output_data) == 0):
            continue

        with open(output_path, 'a', encoding='utf-8') as output_file: # 写入本地文件
            for data in output_data:
                json_line = json.dumps(data, ensure_ascii=False)
                output_file.write(json_line + '\n')
        print("NO." + str(line_cnt) + " " + str( line_cnt/161.1 ) + "%") # 显示当前进度