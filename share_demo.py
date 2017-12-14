from weibo import get_token, post_a_pic


access_token = get_token()
# TODO 弹出对话框 提示用户输入文字
news = input("What's new？\n")
# TODO: pic 待分享图片路径
pic = input("Pic:\n")

post_a_pic(access_token, news, pic)
