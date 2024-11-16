from PIL import Image
import os

def resize_images(input_folder, output_folder, new_size):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        try:
            # Open the image
            with Image.open(os.path.join(input_folder, filename)) as img:
                # Resize the image
                img = img.convert("L")
                img_resized = img.resize(new_size)

                for i in range(3):
                    for j in range(3):
                        # Crop and save each sub-image
                        box = (j * 128, i * 128, (j + 1) * 128, (i + 1) * 128)
                        sub_img = img_resized.crop(box)
                        sub_img_filename = f"{os.path.splitext(filename)[0]}_{i}{j}.png"  # Change extension if needed
                        sub_img.save(os.path.join(output_folder, sub_img_filename))
                        print(f"Saved {sub_img_filename} to {output_folder}")
                # Save the resized image to the output folder
                #img_resized.save(os.path.join(output_folder, filename))
                #print(f"Resized {filename} and saved to {output_folder}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    # Input and output folders
    input_folder = "data_raw/data_raw/fronts/train"
    output_folder = "128_patches/fronts_256/train"
    
    # New size for the images (width, height)
    new_size = (256, 256)
    
    # Call the function to resize images
    resize_images(input_folder, output_folder, new_size)
