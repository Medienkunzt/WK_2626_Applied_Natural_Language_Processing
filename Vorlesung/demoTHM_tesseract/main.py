import cv2 
import pytesseract

# Path to the location of the Tesseract-OCR executable/command
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

img = cv2.imread('2Architecture_Overview_Tesseract.png')

# Adding custom options
custom_config = r'--oem 3 --psm 6'
output = pytesseract.image_to_string(img, config=custom_config)
# show output
print(output)

# write outputfile
file = open("results.txt", "a")
file.write(output)
file.close()


