# -*- coding: utf-8 -*-
"""
https://oai.azure.com/portal/be5567c3dd4d49eb93f58914cccf3f02/deployment
clausa gpt4
"""

import time
import requests
import config

def chatgpt(prompt, temperature=0.7, n=1, top_p=1, max_tokens=4000, timeout=60):
    messages = [{"role": "user", "content": prompt}]
    payload = {
        "messages": messages,
        "model": "gpt-3.5-turbo",
        "temperature": temperature,
        "n": n,
        "top_p": top_p,
        "max_tokens": max_tokens
    }
    retries = 0
    while True:
        try:
            r = requests.post('https://api.openai.com/v1/chat/completions',
                headers = {
                    "Authorization": f"Bearer {config.OPENAI_KEY}",
                    "Content-Type": "application/json"
                },
                json = payload,
                timeout=timeout
            )
            if r.status_code != 200:
                retries += 1
                time.sleep(1)
                if retries >= 3:
                    with open('error-log.txt', 'a', encoding='utf-8') as outf:
                        outf.write(f"\n\n{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
                        outf.write(f"response:{r}")
                        outf.write(f"prompt:\n{prompt}")
                    exit()
            else:
                break
        except requests.exceptions.ReadTimeout:
            time.sleep(1)
            retries += 1
            if retries >= 3:
                with open('error-log.txt', 'a', encoding='utf-8') as outf:
                    outf.write(f"\n\n{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
                    outf.write(f"超时未响应！")
                    outf.write(f"prompt:\n{prompt}")
                exit()
    r = r.json()
    return [choice['message']['content'] for choice in r['choices']]



if __name__ == '__main__':
    print("你好")
    r = chatgpt("hello！")
    print("你好")



