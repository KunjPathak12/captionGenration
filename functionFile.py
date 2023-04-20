from initialFile import *

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def imgPredictStep(imagePaths):
    images =[]
    for imgPath in imagePaths:
        selectedImage = Image.open(imgPath)
        # let's check if it's rgb or not if it's not let's convert it to one!
        if selectedImage.mode !="RGB":
            selectedImage = selectedImage.convert(mode="RGB")
        images.append(selectedImage)
    pixel_values = feature_extractor(images = images, return_tensors="pt").pixel_values
    # print(type(pixel_values),pixel_values)
    pixel_values = pixel_values.to(device)
    outputIds = model.generate(pixel_values, **gen_kwargs)

    captions = tokenizer.batch_decode(outputIds, skip_special_tokens=True)
    captions = [caption.strip() for caption in captions]
    return captions