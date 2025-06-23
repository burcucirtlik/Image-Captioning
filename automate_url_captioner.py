import requests
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
from transformers import AutoProcessor, BlipForConditionalGeneration

# Load processors and models
processors = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# URl of the page to scrape
url = "https://en.wikipedia.org/wiki/IBM"

#Download the Page
response = requests.get(url)

# Parse the page with beautifulsoup
soup = BeautifulSoup(response.text, "html.parser")

# Find all the Images
img_elements = soup.find_all('img')

# Open a file to write the captions
with open("captions.txt", "w") as caption_file:
    for img_element in img_elements:
        img_url = img_element.get('src')

        # Skip if the image is an SVG or too small (likely an icon)
        if 'svg' in img_url or '1x1' in img_url:
            continue
        
        # Correct the URL if it's malformed
        if img_url.startswith('//'):
            img_url = 'https:' +img_url
        elif not img_url.startswith('http://') and not img_url.startswith('https://'): 
            continue

        try:
            #Download the Image
            response = requests.get(img_url)

            #Convert img data to PIL Image
            raw_image= Image.open(BytesIO(response.content))

            if raw_image.size[0] * raw_image.size[1] < 400:
                continue
            
            raw_image = raw_image.convert('RGB')

            #Process the Image
            inputs = processors(raw_image, return_tensors="pt")

            # Generate a caption for the image
            out = model.generate(**inputs, max_new_tokens=50)

            # Decode the generated tokens to text
            caption = processors.decode(out[0], skip_special_tokens=True)

            # Write the caption to the file, prepended by the image URL
            caption_file.write(f"{img_url}: {caption}\n")
        except Exception as e:
            print(f"Error Processing image {img_url}: {e}")
            continue

    

