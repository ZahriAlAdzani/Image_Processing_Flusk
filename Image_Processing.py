from PIL import Image
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from collections import Counter
import os

import matplotlib
import cv2
matplotlib.use('Agg')

save_dir = "static/images"  # Change this directory to match your HTML


def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def is_grayscale(img_path):
    img = Image.open(img_path)
    return img.mode == 'L'  # Check if the image mode is grayscale


def is_rgba(img_path):
    img = Image.open(img_path)
    return img.mode == 'RGBA'  # Check if the image mode is RGBA


def grayscale(img_path):
    convert_to_grayscale(img_path, save_dir)
    generate_histogram(f"{save_dir}/img_grayscale.jpg", save_dir)


def both_histogram(img_path):
    convert_to_grayscale(img_path, 'static/temp')
    generate_histogram('static/temp/img_grayscale.jpg', 'static/images/')

    if (not is_grayscale(img_path)):
        histogram_rgb('static/images/img_now.jpg')


def histogram_rgb(img_path):
    img = Image.open(img_path)
    img_arr = np.asarray(img)

    r = img_arr[:, :, 0].flatten()
    g = img_arr[:, :, 1].flatten()
    b = img_arr[:, :, 2].flatten()
    data_r = Counter(r)
    data_g = Counter(g)
    data_b = Counter(b)
    data_rgb = [data_r, data_g, data_b]
    colors = ['red', 'green', 'blue']
    data_hist = list(zip(colors, data_rgb))
    ensure_directory_exists(save_dir)

    # Check if the image is RGBA or PNG and save it as JPG if needed
    if img.mode == 'RGBA' or img.format == 'PNG':
        img = img.convert('RGB')
        img.save(f"{save_dir}/img_now.jpg", "JPEG")
    else:
        img.save(f"{save_dir}/img_now.jpg")

    for data in data_hist:
        plt.bar(list(data[1].keys()), data[1].values(), color=f"{data[0]}")
        plt.savefig(f"{save_dir}/{data[0]}_histogram.jpg", dpi=300)
        plt.clf()
    return True


def df(img):  # to make a histogram (count distribution frequency)
    values = [0]*256
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            values[img[i, j]] += 1
    return values


def cdf(hist):  # cumulative distribution frequency
    cdf = [0] * len(hist)  # len(hist) is 256
    cdf[0] = hist[0]
    for i in range(1, len(hist)):
        cdf[i] = cdf[i-1]+hist[i]
    # Now we normalize the histogram
    # What your function h was doing before
    cdf = [ele*255/cdf[-1] for ele in cdf]
    return cdf


def histogram_equalizer():
    img = cv2.imread('static\images\img_now.jpg', 0)
    my_cdf = cdf(df(img))
    # use linear interpolation of cdf to find new pixel values. Scipy alternative exists
    image_equalized = np.interp(img, range(0, 256), my_cdf)
    cv2.imwrite('static/images/img_now.jpg', image_equalized)


def convert_to_grayscale(img_path, save_dir):
    img = Image.open(img_path)
    img_arr = np.asarray(img)

    if is_grayscale(img_path):
        # If the image is already grayscale, just save it without modification
        ensure_directory_exists(save_dir)
        img.save(f"{save_dir}/img_grayscale.jpg")
    else:
        if is_rgba(img_path):
            r = img_arr[:, :, 0]
            g = img_arr[:, :, 1]
            b = img_arr[:, :, 2]
            a = img_arr[:, :, 3]
            # Convert RGBA to grayscale by weighting R, G, and B channels and ignoring alpha (transparency) channel
            new_arr = (r * 0.2989 + g * 0.5870 + b * 0.1140).astype('uint8')
            new_img = Image.fromarray(new_arr)
            new_img.save(f"{save_dir}/img_grayscale.jpg")
        else:
            r = img_arr[:, :, 0]
            g = img_arr[:, :, 1]
            b = img_arr[:, :, 2]
            new_arr = (r.astype(int) + g.astype(int) + b.astype(int)) // 3
            new_arr = new_arr.astype('uint8')
            new_img = Image.fromarray(new_arr)
            new_img.save(f"{save_dir}/img_grayscale.jpg")


def generate_histogram(img_path, save_dir):
    img = Image.open(img_path)
    img_arr = np.asarray(img)
    g = img_arr[:, :].flatten()
    data_g = Counter(g)
    plt.bar(list(data_g.keys()), data_g.values(), color='black')
    ensure_directory_exists(save_dir)
    plt.savefig(f"{save_dir}/grey_histogram.jpg", dpi=300)
    plt.clf()


def apply_edge_detection(img_path):
    img = Image.open(img_path)
    img_arr = np.asarray(img)

    # Perform edge detection using the Canny algorithm from OpenCV
    # You can adjust the threshold values as needed
    edges = cv2.Canny(img_arr, 100, 200)

    # Convert the edge-detected image array back to a PIL image
    edge_img = Image.fromarray(edges)

    # Save the edge-detected image
    ensure_directory_exists(save_dir)
    edge_img.save(f"{save_dir}/img_now.jpg")

    return True


def apply_image_sharpening(img_path):
    img = Image.open(img_path)

    # Apply image sharpening using the PIL ImageFilter.SHARPEN filter
    sharpened_img = img.filter(ImageFilter.SHARPEN)

    # Save the sharpened image
    ensure_directory_exists(save_dir)
    sharpened_img.save(f"{save_dir}/img_now.jpg")

    return True


def apply_blur(img_path):
    img = Image.open(img_path)

    # Apply blur using the PIL ImageFilter.BLUR filter
    blurred_img = img.filter(ImageFilter.BLUR)

    # Save the blurred image
    ensure_directory_exists(save_dir)
    blurred_img.save(f"{save_dir}/img_now.jpg")

    return True


def normal(img_path):
    img = Image.open(img_path)
    img.save(f"{save_dir}/img_now.jpg")


def move_left(img_path, pixels):
    img = Image.open(img_path)
    width, height = img.size

    # Create a blank image with black pixels
    new_img = Image.new('RGB', (width, height), (0, 0, 0))

    # Paste the cropped portion onto the new image
    new_img.paste(img.crop((pixels, 0, width, height)), (0, 0))

    new_img.save(f"{save_dir}/img_now.jpg")
    return True


def move_right(img_path, pixels):
    img = Image.open(img_path)
    width, height = img.size

    # Create a blank image with black pixels
    new_img = Image.new('RGB', (width, height), (0, 0, 0))

    # Paste the cropped portion onto the new image
    new_img.paste(img.crop((0, 0, width - pixels, height)), (pixels, 0))

    new_img.save(f"{save_dir}/img_now.jpg")
    return True


def move_up(img_path, pixels):
    img = Image.open(img_path)
    width, height = img.size

    # Create a blank image with black pixels
    new_img = Image.new('RGB', (width, height), (0, 0, 0))

    # Paste the cropped portion onto the new image
    new_img.paste(img.crop((0, pixels, width, height)), (0, 0))

    new_img.save(f"{save_dir}/img_now.jpg")
    return True


def move_down(img_path, pixels):
    img = Image.open(img_path)
    width, height = img.size

    # Create a blank image with black pixels
    new_img = Image.new('RGB', (width, height), (0, 0, 0))

    # Paste the cropped portion onto the new image
    new_img.paste(img.crop((0, 0, width, height - pixels)), (0, pixels))

    new_img.save(f"{save_dir}/img_now.jpg")
    return True


def zoom_in(img_path, factor):
    img = Image.open(img_path)
    img = img.convert("RGB")
    img_arr = np.asarray(img)
    height, width, channels = img_arr.shape
    new_height = int(height * factor)
    new_width = int(width * factor)

    # Create a new array filled with black pixels
    new_arr = np.zeros((new_height, new_width, channels), dtype=np.uint8)

    for i in range(new_height):
        for j in range(new_width):
            # Calculate the corresponding pixel coordinates in the original image
            original_i = int(i // factor)
            original_j = int(j // factor)
            new_arr[i, j] = img_arr[original_i, original_j]

    new_img = Image.fromarray(new_arr)
    new_img.save("static/images/img_now.jpg")


def zoom_out(img_path, factor):
    img = Image.open(img_path)
    img = img.convert("RGB")
    img_arr = np.asarray(img)
    height, width, channels = img_arr.shape
    new_height = int(height // factor)
    new_width = int(width // factor)

    # Create a new array filled with black pixels
    new_arr = np.zeros((new_height, new_width, channels), dtype=np.uint8)

    for i in range(new_height):
        for j in range(new_width):
            r_sum, g_sum, b_sum = 0, 0, 0
            # Calculate the corresponding pixel coordinates in the original image
            for x in range(int(factor)):
                for y in range(int(factor)):
                    original_i = int(i * factor + x)
                    original_j = int(j * factor + y)
                    pixel_value = img_arr[original_i, original_j]
                    r_sum += pixel_value[0]
                    g_sum += pixel_value[1]
                    b_sum += pixel_value[2]
            new_arr[i, j] = (int(r_sum // (factor ** 2)), int(g_sum //
                             (factor ** 2)), int(b_sum // (factor ** 2)))

    new_img = Image.fromarray(new_arr)
    new_img.save("static/images/img_now.jpg")


def histogram_specification_Done(image, reference_hist):
    # Buat output citra kosong
    output_image = np.zeros_like(image)

    # Kalkulasi CDF citra kedua
    reference_cdf = np.cumsum(reference_hist)

    # Kalkulasi CDF input citra histogram
    input_cdf = np.cumsum(cv2.calcHist([image], [0], None, [256], [0, 256]))

    # Normalisasi CDF agar valuesnya range [0, 255]
    reference_cdf = ((reference_cdf - reference_cdf.min()) *
                     255) / (reference_cdf.max() - reference_cdf.min())
    input_cdf = ((input_cdf - input_cdf.min()) * 255) / \
        (input_cdf.max() - input_cdf.min())

    # Memetakan intensitas citra masukan ke CDF citra kedua
    output_image = cv2.LUT(image, reference_cdf.astype('uint8'))

    return output_image


def histogram_specification(source_img_path):
    # Load the source and target images
    # Convert the images to grayscale
    convert_to_grayscale(source_img_path, 'static/temp')

    source_img = cv2.imread(
        'static/temp/img_grayscale.jpg', 0)  # Load as grayscale
    target_img = cv2.imread(
        'static/images/img_grayscale.jpg', 0)  # Load as grayscale
    gray_target_hist = cv2.calcHist([source_img], [0], None, [256], [0, 256])

    target_img = histogram_specification_Done(target_img, gray_target_hist)

    # Save the target image
    ensure_directory_exists(save_dir)
    cv2.imwrite(f"{save_dir}/img_now.jpg", target_img)


def brightness_addition():
    img = Image.open("static/images/img_now.jpg")
    img_arr = np.asarray(img).astype('uint16')
    img_arr = img_arr+100
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/images/img_now.jpg")


def brightness_substraction():
    img = Image.open("static/images/img_now.jpg")
    img_arr = np.asarray(img).astype('int16')
    img_arr = img_arr-100
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/images/img_now.jpg")


def brightness_multiplication():
    img = Image.open("static/images/img_now.jpg")
    img_arr = np.asarray(img)
    img_arr = img_arr*1.25
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/images/img_now.jpg")


def brightness_division():
    img = Image.open("static/images/img_now.jpg")
    img_arr = np.asarray(img)
    img_arr = img_arr/1.25
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/images/img_now.jpg")


def threshold(lower_thres, upper_thres):
    # Load the image
    img = Image.open("static/images/img_now.jpg")

    # Convert the image to a NumPy array
    img_arr = np.array(img)

    # Create a condition for pixels within the specified threshold range
    condition = np.logical_and(np.greater_equal(img_arr, lower_thres),
                               np.less_equal(img_arr, upper_thres))

    # Replace pixel values outside the threshold range with 255
    img_arr[~condition] = 255

    # Create a new PIL Image from the modified NumPy array
    new_img = Image.fromarray(img_arr)

    # Save the new image
    new_img.save("static/images/img_now.jpg")
