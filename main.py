import os
import pytesseract
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from preprocessing_pipeline import preprocess_pipeline

def ocr_image(image):
    """
    Perform OCR on the given image and return the extracted text.
    """
    text = pytesseract.image_to_string(image)
    return text

def summarize_text(text):
    """
    Summarize the given text using a transformer model.
    """
    model_name = "t5-base"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Initialize the summarization pipeline
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    # summarizer = pipeline("summarization")
    summary = summarizer(text, max_length=350, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def main(image_path):
    # Preprocess the image
    preprocessed_image = preprocess_pipeline(image_path)
    
    # Perform OCR to extract text
    extracted_text = ocr_image(preprocessed_image)
    print("Extracted Text:", extracted_text)
    
    # Summarize the extracted text
    summary = summarize_text(extracted_text)
    print("Summary:", summary)
    
    return summary

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, 'data', 'one_column.jpeg')
    
    main(image_path)
