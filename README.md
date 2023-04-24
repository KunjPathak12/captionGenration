# captionGenration from Image with PyTorch,huggingface and encoder-decoder Transformer models
This code utilizes the Hugging Face Transformers library to predict image captions using a pre-trained VisionEncoderDecoderModel that combines a Vision Transformer (ViT) and a GPT-2 language model. The ViTImageProcessor is used to extract features from the image, and the AutoTokenizer is used to tokenize the captions.

The function predict_step takes a list of image paths as input, opens and converts each image to RGB mode, extracts the pixel values using the feature_extractor, generates a caption using the pre-trained model, and returns a list of predicted captions.

The VisionEncoderDecoderModel class is used to load and use the pre-trained vision encoder-decoder model.
The ViTImageProcessor class is used to extract image features from an input image.
The AutoTokenizer class is used to tokenize the captions.
The torch module is used for tensor operations and device management.
The PIL module is used for image processing.
