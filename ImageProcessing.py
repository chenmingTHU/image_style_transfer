from PIL import Image, ImageEnhance, ImageDraw, ImageFont


# 比例 a.1:1 b.4:3 c.3:4 d.16:9 e.9:16
def im_resize(img, flag):
    # flag a.1:1 b.4:3 c.3:4 d.16:9 e.9:16
    height, width = img.size
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

    return dst


# 反转 左右/上下
# dst = img.transpose(Image.FLIP_LEFT_RIGHT)
# dst = img.transpose(Image.FLIP_TOP_BOTTOM)


# 旋转 顺/逆时针
# dst = img.transpose(Image.ROTATE_270)
# dst = img.transpose(Image.ROTATE_90)


# 亮度、锐利度、对比度
# dst = ImageEnhance.Brightness(img)
# dst = ImageEnhance.Sharpness(img)
# dst = ImageEnhance.Contrast(img)
# dst = dst.enhance(factor)


# 添加水印
# flag 0:black 1:white
def watermark(image, texture="by 404", flag=0, x=60, y=20, size=20):
    img = image.convert('RGBA')
    txt = Image.new('RGBA', img.size, (0, 0, 0, 0))
    fnt = ImageFont.truetype("C:\Windows\Fonts\FTLTLT.TTF", size)
    dst = ImageDraw.Draw(txt)
    if flag == 0:
        dst.text((txt.size[0] - x, txt.size[1] - y), texture, font=fnt, fill=(0, 0, 0, 255))
    else:
        dst.text((txt.size[0] - x, txt.size[1] - y), texture, font=fnt, fill=(255, 255, 255, 255))
    dst = Image.alpha_composite(img, txt)
    return dst


# 添加白色边框
def border(img, factor=0.02):
    w, h = img.size
    dst = ImageDraw.Draw(img)
    w_half = int(w*factor)
    h_half = int(h*factor)
    dst.line(((0, h_half-1), (w-1, h_half-1)), fill=(255, 255, 255), width=2*h_half)
    dst.line(((0, h-1-h_half), (w-1, h-1-h_half)), fill=(255, 255, 255), width=2*h_half)
    dst.line(((w_half-1, 0), (w_half-1, h-1)), fill=(255, 255, 255), width=2*w_half)
    dst.line(((w-1-w_half, 0), (w-1-w_half, h-1)), fill=(255, 255, 255), width=2*w_half)
    return img

