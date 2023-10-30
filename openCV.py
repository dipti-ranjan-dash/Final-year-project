import cv2
import numpy as np
import os
import pandas as pd

def hist_ncc(image_path1, image_path2):
    # Load the two images you want to compare
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)

    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    # Convert the images to grayscale (assuming they are not grayscale already)
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Compute histograms for the two images
    hist_image1 = cv2.calcHist([gray_image1], [0], None, [256], [0, 256])
    hist_image2 = cv2.calcHist([gray_image2], [0], None, [256], [0, 256])

    # Normalize the histograms
    hist_image1 /= hist_image1.sum()
    hist_image2 /= hist_image2.sum()

    # Calculate histogram comparison values using different methods
    hist_intersection = cv2.compareHist(hist_image1, hist_image2, cv2.HISTCMP_INTERSECT)
    hist_correlation = cv2.compareHist(hist_image1, hist_image2, cv2.HISTCMP_CORREL)
    hist_chi_square = cv2.compareHist(hist_image1, hist_image2, cv2.HISTCMP_CHISQR)
    hist_bhattacharyya = cv2.compareHist(hist_image1, hist_image2, cv2.HISTCMP_BHATTACHARYYA)

    # Calculate Normalized Cross-Correlation (NCC)
    ncc = np.sum((gray_image1 - np.mean(gray_image1)) * (gray_image2 - np.mean(gray_image2))) / (
                (np.std(gray_image1) * np.std(gray_image2)) * gray_image1.size)

    # Calculate the final histogram comparison value as the average of all values
    comparison_values = [hist_intersection, hist_correlation, hist_chi_square, hist_bhattacharyya, ncc]
    final_comparison_value = sum(comparison_values) / len(comparison_values)
    return final_comparison_value

# Input folders for images
input_folder = r'C:\Users\soham\python prog\image_pros\input'
comparison_folder = r'C:\Users\soham\python prog\image_pros\comp'

# Create an empty list to store the results
results = []

# Iterate through images in the input folder and the comparison folder
for filename1 in os.listdir(input):
    for filename_comp in os.listdir(comp):
        image_path1 = os.path.join(input, filename1)
        image_path_comp = os.path.join(comp, filename_comp)

        # Calculate the comparison value
        result = hist_ncc(image_path1, image_path_comp)

        # Append the results to the list
        results.append({"Image1": filename1, "Comparison_Image": filename_comp, "Comparison_Value": result})

# Create a DataFrame from the results list
results_df = pd.DataFrame(results)

# Save the results to an Excel file
output_excel_file = "image_comparisons_OR_007.xlsx"
results_df.to_excel(output_excel_file, index=False)

print(f"Comparison results saved to {output_excel_file}")
