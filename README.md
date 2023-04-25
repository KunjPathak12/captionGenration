# captionGenration from Image with PyTorch,huggingface and encoder-decoder Transformer models

There are three files present i the repository namely driver, functionFile and the Initialfile respectively.

The driver is the file which calls the function from the functionFile and import all the essentials files from the prior files.
Hence if you want to use this project for caption generation of your image just give your image fie's path into driver file and run that file after instaling the required modules and dependencies mentioned.

The functionFile contains the function created that utilizes the modules imported and make use of various mehtods to extract the feature from the image and 
This code utilizes the Hugging Face Transformers library to predict image captions using a pre-trained VisionEncoderDecoderModel that combines a Vision Transformer (ViT) and a GPT-2 language model. The ViTImageProcessor is used to extract features from the image, and the AutoTokenizer is used to tokenize the captions.

The function predict_step takes a list of image paths as input, opens and converts each image to RGB mode, extracts the pixel values using the feature_extractor, generates a caption using the pre-trained model, and returns a list of predicted captions.

The VisionEncoderDecoderModel class is used to load and use the pre-trained vision encoder-decoder model.
The ViTImageProcessor class is used to extract image features from an input image.
The AutoTokenizer class is used to tokenize the captions.
The torch module is used for tensor operations and device management.
The PIL module is used for image processing.
