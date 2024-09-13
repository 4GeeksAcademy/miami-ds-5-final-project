from PIL import Image

# Open the WEBP image
webp_image = Image.open('src/static/uploads/cancer.webp')

# Convert the image to RGBA (to handle transparency)
rgba_image = webp_image.convert("RGBA")

# Make the background transparent
# In this example, we'll assume the background is white (255, 255, 255)
# You can change the color to whatever the background color is in the WEBP
data = rgba_image.getdata()

new_data = []
for item in data:
    # Change all white (also includes shades of white) pixels to transparent
    # Here we are assuming the background is pure white (255, 255, 255)
    if item[0] > 100 and item[1] > 100 and item[2] > 100:  # R, G, B values
        # Set transparent for white pixels
        new_data.append((255, 255, 255, 0))  # Set alpha to 0
    else:
        new_data.append(item)  # Keep other pixels unchanged

# Update image data with transparency
rgba_image.putdata(new_data)

# Save the new PNG file
rgba_image.save('src/static/uploads/cancer.png', 'PNG')

print("Conversion complete: WEBP to PNG with transparent background.")