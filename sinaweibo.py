import requests
import webbrowser

client_id = "2407786039"  # App Key
client_secret = ""  # App Secret
url_authorize = "https://api.weibo.com/oauth2/authorize"
url_get_token = "https://api.weibo.com/oauth2/access_token"
redirect_uri = "https://api.weibo.com/oauth2/default.html"

webbrowser.open("{}?client_id={}&redirect_uri={}&response_type=code".format(url_authorize, client_id, redirect_uri), new=0, autoraise=True)

code = input("Input the code\n")

payload = {
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri
             }

r = requests.post(url_get_token, data=payload)

print(r.text)
