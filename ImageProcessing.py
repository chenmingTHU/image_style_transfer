from PIL import Image, ImageEnhance, ImageDraw, ImageFont


# 比例 a.1:1 b.4:3 c.3:4 d.16:9 e.9:16
def img_resize(img_path, flag):
    # flag a.1:1 b.4:3 c.3:4 d.16:9 e.9:16
    img = Image.open(img_path)
    width, height = img.size
    if height <= width:
        short_size = height
        long_size = width
    else:
        short_size = width
        long_size = height

    if flag == 'a':
        dst = img.resize((short_size, short_size), Image.BICUBIC)
    elif flag == 'b':
        dst = img.resize((long_size, int(3/4*long_size)), Image.BICUBIC)
    elif flag == 'c':
        dst = img.resize((int(3/4*long_size), long_size), Image.BICUBIC)
    elif flag == 'd':
        dst = img.resize((long_size, int(9/16*long_size)), Image.BICUBIC)
    elif flag == 'e':
        dst = img.resize(int((9/16*long_size), long_size), Image.BICUBIC)

    dst_path = "temp.jpg"
    dst.save(dst_path)


# 反转 左右/上下
# dst = img.transpose(Image.FLIP_LEFT_RIGHT)
def img_flip_left(img_path, out_path="temp/temp.jpg"):
    img = Image.open(img_path)
    dst = img.transpose(Image.FLIP_LEFT_RIGHT)
    dst_path = out_path
    dst.save(dst_path)


# dst = img.transpose(Image.FLIP_TOP_BOTTOM)
def img_flip_up(img_path, out_path="temp/temp.jpg"):
    img = Image.open(img_path)
    dst = img.transpose(Image.FLIP_TOP_BOTTOM)
    dst_path = out_path
    dst.save(dst_path)


# 旋转 顺/逆时针
# dst = img.transpose(Image.ROTATE_270)
def img_rotate_cw(img_path, out_path="temp/temp.jpg"):
    img = Image.open(img_path)
    dst = img.transpose(Image.ROTATE_270)
    dst_path = out_path
    dst.save(dst_path)


# dst = img.transpose(Image.ROTATE_90)
def img_rotate_ccw(img_path, out_path="temp/temp.jpg"):
    img = Image.open(img_path)
    dst = img.transpose(Image.ROTATE_90)
    dst_path = out_path
    dst.save(dst_path)


# 亮度、锐利度、对比度
# dst = ImageEnhance.Brightness(img)
def img_enhance(img_path, bright, sharp, contrast, out_path="temp/temp.jpg"):
    img = Image.open(img_path)
    dst = ImageEnhance.Brightness(img)
    img = dst.enhance(bright)
    dst = ImageEnhance.Sharpness(img)
    img = dst.enhance(sharp)
    dst = ImageEnhance.Contrast(img)
    dst = dst.enhance(contrast)
    dst_path = out_path
    dst.save(dst_path)

"""
def img_enhance_brightness(img_path, factor, out_path="temp/temp.jpg"):
    img = Image.open(img_path)
    dst = ImageEnhance.Brightness(img)
    dst = dst.enhance(factor)
    dst_path = out_path
    dst.save(dst_path)


# dst = ImageEnhance.Sharpness(img)
def img_enhance_sharpness(img_path, factor, out_path="temp/temp.jpg"):
    img = Image.open(img_path)
    dst = ImageEnhance.Sharpness(img)
    dst = dst.enhance(factor)
    dst_path = out_path
    dst.save(dst_path)


# dst = ImageEnhance.Contrast(img)
def img_enhance_contrast(img_path, factor, out_path="temp/temp.jpg"):
    img = Image.open(img_path)
    dst = ImageEnhance.Contrast(img)
    dst = dst.enhance(factor)
    dst_path = out_path
    dst.save(dst_path)
"""

# 添加水印
def watermark(img_path, texture="by 404", flag=0, x=60, y=20, size=20):
    img = Image.open(img_path)
    img = img.convert('RGBA')
    txt = Image.new('RGBA', img.size, (0, 0, 0, 0))
    fnt = ImageFont.truetype("C:\Windows\Fonts\FTLTLT.TTF", size)
    dst = ImageDraw.Draw(txt)
    if flag == 0:
        dst.text((txt.size[0] - x, txt.size[1] - y), texture, font=fnt, fill=(0, 0, 0, 255))
    else:
        dst.text((txt.size[0] - x, txt.size[1] - y), texture, font=fnt, fill=(255, 255, 255, 255))
    dst = Image.alpha_composite(img, txt)
    dst = dst.convert('RGB')

    dst_path = "temp.jpg"
    dst.save(dst_path)


# 添加白色边框
def border(img_path, factor=0.02):
    img = Image.open(img_path)
    w, h = img.size
    dst = ImageDraw.Draw(img)
    w_half = int(w*factor)
    h_half = int(h*factor)
    dst.line(((0, h_half-1), (w-1, h_half-1)), fill=(255, 255, 255), width=2*h_half)
    dst.line(((0, h-1-h_half), (w-1, h-1-h_half)), fill=(255, 255, 255), width=2*h_half)
    dst.line(((w_half-1, 0), (w_half-1, h-1)), fill=(255, 255, 255), width=2*w_half)
    dst.line(((w-1-w_half, 0), (w-1-w_half, h-1)), fill=(255, 255, 255), width=2*w_half)

    dst_path = "temp.jpg"
    img.save(dst_path)


# 添加水印
# flag 0:black 1:white
# def watermark(image, texture="by 404", flag=0, x=60, y=20, size=20):
#     img = image.convert('RGBA')
#     txt = Image.new('RGBA', img.size, (0, 0, 0, 0))
#     fnt = ImageFont.truetype("C:\Windows\Fonts\FTLTLT.TTF", size)
#     dst = ImageDraw.Draw(txt)
#     if flag == 0:
#         dst.text((txt.size[0] - x, txt.size[1] - y), texture, font=fnt, fill=(0, 0, 0, 255))
#     else:
#         dst.text((txt.size[0] - x, txt.size[1] - y), texture, font=fnt, fill=(255, 255, 255, 255))
#     dst = Image.alpha_composite(img, txt)
#     return dst


# 添加白色边框
# def border(img, factor=0.02):
#     w, h = img.size
#     dst = ImageDraw.Draw(img)
#     w_half = int(w*factor)
#     h_half = int(h*factor)
#     dst.line(((0, h_half-1), (w-1, h_half-1)), fill=(255, 255, 255), width=2*h_half)
#     dst.line(((0, h-1-h_half), (w-1, h-1-h_half)), fill=(255, 255, 255), width=2*h_half)
#     dst.line(((w_half-1, 0), (w_half-1, h-1)), fill=(255, 255, 255), width=2*w_half)
#     dst.line(((w-1-w_half, 0), (w-1-w_half, h-1)), fill=(255, 255, 255), width=2*w_half)
#     return img

