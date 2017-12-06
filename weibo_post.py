import requests
from requests_toolbelt import MultipartEncoder

url_share = "https://api.weibo.com/2/statuses/share.json"
access_token = ""  # access_token got from sinaweibo.py
status = input('What\'s new?\n') + "http://www.baidu.com"
pic = input('picture:\n')

m = MultipartEncoder(
    fields={'access_token': access_token, 'status': status,
            'pic': (pic, open(pic, 'rb'))}
    )

r = requests.post(url_share, data=m, headers={'Content-Type': m.content_type})

print(r.text)
