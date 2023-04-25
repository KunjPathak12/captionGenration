# captionGenration from Image with PyTorch,huggingface and encoder-decoder Transformer models

There are three files present i the repository namely driver, functionFile and the Initialfile respectively.

The driver is the file which calls the function from the functionFile and import all the essentials files from the prior files.
Hence if you want to use this project for caption generation of your image just give your image fie's path into driver file and run that file after instaling the required modules and dependencies mentioned.

The functionFile contains the function developed to make use of the imported modules and multiple means to extract features from the image and produce the caption. multiple supporting mechanisms and methods from various modules are used to produce captions from the provided image.
Using a pre-trained VisionEncoderDecoderModel that combines a Vision Transformer (ViT) and a GPT-2 language model, this code makes use of the Hugging Face Transformers library to predict image captions. The image's characteristics are extracted using the ViTImageProcessor, and the captions are tokenized using the AutoTokenizer.The predict_step function takes a list of image paths as input, opens each image, changes it to RGB mode, uses feature_extractor to extract the pixel values, and then uses the pre-trained model to output a list of predicted captions.

The pre-trained vision encoder-decoder model is loaded and used by the VisionEncoderDecoderModel class. When an image is input, image features are extracted using the ViTImageProcessor class. The captions are tokenized using the AutoTokenizer class, device management and tensor operations are done with the torch module. Image processing is done using the PIL module.
