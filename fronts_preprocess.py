from PIL import Image
import os

def convert_to_custom_grayscale(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        try:
            # Open the image
            with Image.open(os.path.join(input_folder, filename)) as img:
                # Convert image to grayscale
                img_gray = img.convert("L")

                # Manipulate pixel values
                img_array = img_gray.load()
                width, height = img_gray.size
                for y in range(height):
                    for x in range(width):
                        pixel_value = img_array[x, y]
                        if pixel_value > 0:
                            img_array[x, y] = 255
                # Save the manipulated image
                img_gray.save(os.path.join(output_folder, filename))
                print(f"Converted and saved {filename} to {output_folder}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    # Input and output folders
    input_folder = "128_patches/fronts/test"
    output_folder = "128_patches/fronts_processed/test"
    
    # Call the function to convert to custom grayscale
    convert_to_custom_grayscale(input_folder, output_folder)
