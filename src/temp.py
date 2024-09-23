from PIL import Image

for i in ['cell', 'cancer']:
    webp_image = Image.open(f'src/static/uploads/{i}.webp')
    rgba_image = webp_image.convert("RGBA")
    data = rgba_image.getdata()
    new_data = []
    for item in data:
        if item[0] > 100 and item[1] > 100 and item[2] > 100:
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(item)
    rgba_image.putdata(new_data)
    rgba_image.save(f'src/static/uploads/{i}.png', 'PNG')

    print("Conversion complete: WEBP to PNG with transparent background.")