# scanned-doc-summary-extraction
This repository provides a pipeline for extracting and summarizing text from scanned documents using Tesseract OCR and OpenCV. It integrates preprocessing techniques to enhance image quality for OCR and uses advanced NLP models to generate summaries of the extracted text.

## Dependencies & Installation

- **Pillow:** Used to load images into memory as sequences of numerical arrays (pixels).
- **OpenCV:** Used for image manipulation.
- **Tesseract:** OCR engine for recognizing text in images.
- **Transformers:** Used for text summarization using models like BERT or GPT-2. 
- **TensorFlow-Keras Compatibility:** Install for backward compatibility with Keras.

  
## Project Structure

The repository contains the following:

- **Experiments folder:** Contains Jupyter notebooks used for experimentation and testing.
- **main.py:** Contains the final product.

## Current Challenges

- **Image Preprocessing:** The project is still facing challenges in implementing preprocessing techniques that effectively handle variations in scanned document qualities. This includes factors such as image resolution, noise levels, and text orientation.



