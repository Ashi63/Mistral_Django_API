import cv2
import pytesseract
import os


# Path for Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def process_image_with_word_bounding_boxes(image_path, processed_folder='processed'):
    """
    Process an image, detect text, draw word-level bounding boxes around the text, and save the processed image.

    Args:
        image_path (str): Path to the input image file.
        processed_folder (str): Path to the folder where the processed image will be saved. Default is 'processed'.
    """
    # Create the processed folder if it doesn't exist
    if not os.path.exists(processed_folder):
        os.makedirs(processed_folder)

    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use Pytesseract to extract text from the image
    data = pytesseract.image_to_data(gray_image, output_type=pytesseract.Output.DICT)

    # Iterate over each word detected
    for i in range(len(data['text'])):
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        # Draw bounding box if the word is not empty
        if data['text'][i].strip() != '':
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)

    # Save the image with word-level bounding boxes in the processed folder
    processed_image_path = os.path.join(processed_folder, 'processed_image_with_word_boxes.jpg')
    cv2.imwrite(processed_image_path, image)

    print(f"Processed image with word-level bounding boxes saved at {processed_image_path}")

# Example usage:
#image_path = r"C:\Users\Alkashi\Desktop\Botmatic\Data_Sample\Data\Invoices\invoice.png"
#process_image_with_word_bounding_boxes(image_path)
