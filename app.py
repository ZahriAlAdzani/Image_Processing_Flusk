# web-app for API image manipulation

from flask import Flask, request, render_template, send_from_directory
import os
from PIL import Image
import Image_Processing
import threading
from collections import Counter
import random


app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


# default access page
@app.route("/")
def main():
    return render_template('index.html')


# upload selected image and forward to processing page

def process_image(destination):
    status1 = Image_Processing.grayscale(destination)
    status2 = Image_Processing.histogram_rgb(destination)
    if status1 and status2:
        print("Image processing completed.")

# Custom Jinja2 filter to zip lists


def zip_lists(a, b, c):
    return zip(a, b, c)


app.jinja_env.filters['zip_lists'] = zip_lists


@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'static/images/')

    # create image directory if not found
    if not os.path.isdir(target):
        os.mkdir(target)

    # retrieve file from html file-picker
    upload = request.files.getlist("file")[0]
    print("File name: {}".format(upload.filename))
    filename = upload.filename

    # file support verification
    ext = os.path.splitext(filename)[1]
    if (ext == ".jpg") or (ext == ".png") or (ext == ".bmp") or (ext == ".jpeg"):
        print("File accepted")
    else:
        return render_template("error.html", message="The selected file is not supported"), 400

    # save file
    destination = "/".join([target, filename])
    print("File saved to to:", destination)
    upload.save(destination)

    # wait the process till done

    processing_thread = threading.Thread(target=process_image(destination))
    processing_thread.start()

    # Wait for the processing thread to finish
    processing_thread.join()

    # Open the uploaded image using PIL
    with Image.open(destination) as img:
        width, height = img.size

    df = Image_Processing.table_Data()
    # Pass the image dimensions to the HTML template
    return render_template("processing.html", image_name=filename, image_width=width, image_height=height, table=df.to_html(classes='table table-striped'))


# retrieve file from 'static/images' directory
@app.route('/static/images/<filename>')
def send_image(filename):
    return send_from_directory("static/images", filename)


@app.route("/equalization", methods=["POST"])
def equalization():
    filename = request.form['image']
    Image_Processing.histogram_equalizer()
    processing_thread = threading.Thread(
        target=Image_Processing.both_histogram('static/images/img_now.jpg'))
    processing_thread.start()

    # Wait for the processing thread to finish
    processing_thread.join()
    return render_template("processing.html", image_name=filename)


@app.route("/edge_detection", methods=["POST"])
def edge_detection():
    filename = request.form['image']

    # Perform edge detection here (call the appropriate function)
    Image_Processing.apply_edge_detection('static/images/img_now.jpg')

    processing_thread = threading.Thread(
        target=Image_Processing.both_histogram('static/images/img_now.jpg'))
    processing_thread.start()

    # Wait for the processing thread to finish
    processing_thread.join()

    return render_template("processing.html", image_name=filename)


@app.route("/image_sharpening", methods=["POST"])
def image_sharpening():
    filename = request.form['image']

    # Perform image sharpening here (call the appropriate function)
    Image_Processing.apply_image_sharpening('static/images/img_now.jpg')

    processing_thread = threading.Thread(
        target=Image_Processing.both_histogram('static/images/img_now.jpg'))
    processing_thread.start()

    # Wait for the processing thread to finish
    processing_thread.join()

    return render_template("processing.html", image_name=filename)


@app.route("/blur", methods=["POST"])
def blur():
    filename = request.form['image']

    # Perform blur here (call the appropriate function)
    Image_Processing.apply_blur('static/images/img_now.jpg')

    processing_thread = threading.Thread(
        target=Image_Processing.both_histogram('static/images/img_now.jpg'))
    processing_thread.start()

    # Wait for the processing thread to finish
    processing_thread.join()

    return render_template("processing.html", image_name=filename)


@app.route("/normal", methods=["POST"])
def normal():
    filename = request.form['image']

    target = os.path.join(APP_ROOT, 'static/images')
    destination = "/".join([target, filename])

    processing_thread = threading.Thread(
        target=Image_Processing.normal(destination))
    processing_thread.start()

    # Wait for the processing thread to finish
    processing_thread.join()

    processing_thread = threading.Thread(target=process_image(destination))
    processing_thread.start()

    # Wait for the processing thread to finish
    processing_thread.join()

    return render_template("processing.html", image_name=filename)

# Route for the Move Left form


@app.route("/move_left", methods=["POST"])
def move_left():
    filename = request.form['image']
    pixels = int(request.form['pixels'])

    # Perform move left (call the appropriate function)
    Image_Processing.move_left('static/images/img_now.jpg', pixels)

    return render_template("processing.html", image_name=filename)

# Route for the Move Right form


@app.route("/move_right", methods=["POST"])
def move_right():
    filename = request.form['image']
    pixels = int(request.form['pixels'])

    # Perform move right (call the appropriate function)
    Image_Processing.move_right('static/images/img_now.jpg', pixels)

    return render_template("processing.html", image_name=filename)

# Route for the Move Up form


@app.route("/move_up", methods=["POST"])
def move_up():
    filename = request.form['image']
    pixels = int(request.form['pixels'])

    # Perform move up (call the appropriate function)
    Image_Processing.move_up('static/images/img_now.jpg', pixels)

    return render_template("processing.html", image_name=filename)

# Route for the Move Down form


@app.route("/move_down", methods=["POST"])
def move_down():
    filename = request.form['image']
    pixels = int(request.form['pixels'])

    # Perform move down (call the appropriate function)
    Image_Processing.move_down('static/images/img_now.jpg', pixels)

    return render_template("processing.html", image_name=filename)

# Route for the Zoom In form


@app.route("/zoom_in", methods=["POST"])
def zoom_in():
    filename = request.form['image']
    factor = float(request.form['factor'])

    # Perform zoom in (call the appropriate function)
    Image_Processing.zoom_in('static/images/img_now.jpg', factor)

    return render_template("processing.html", image_name=filename)

# Route for the Zoom Out form


@app.route("/zoom_out", methods=["POST"])
def zoom_out():
    filename = request.form['image']
    factor = float(request.form['factor'])

    # Perform zoom out (call the appropriate function)
    Image_Processing.zoom_out('static/images/img_now.jpg', factor)

    return render_template("processing.html", image_name=filename)


@app.route("/histogram_specification", methods=["POST"])
def histogram_specification():
    # Retrieve the uploaded second image
    filename = request.form['image']
    second_image = request.files.get("second_image")
    # save file
    target = os.path.join(APP_ROOT, 'static/images/')
    # Specify the filename here
    destination = os.path.join(target, 'second_image.jpg')
    print("File saved to:", destination)
    second_image.save(destination)

    if second_image:

        # Perform histogram specification using the second image
        Image_Processing.histogram_specification(destination)

        # Start a thread to process the image (if needed)
        processing_thread = threading.Thread(
            target=Image_Processing.both_histogram('static/images/img_now.jpg'))
        processing_thread.start()
        processing_thread.join()

        # Redirect to a page or template after processing
        return render_template("processing.html", image_name=filename)

    # Handle the case where the second image was not provided
    return render_template("error.html", message="Please upload a second image for histogram specification"), 400


@app.route("/blend", methods=["POST"])
def blend():
    # Retrieve parameters from the HTML form
    alpha = request.form['alpha']
    image_name = request.form['image']

    # Get the paths to the target and source images
    target_folder = os.path.join(APP_ROOT, 'static/images')
    source_image_path = os.path.join(target_folder, 'img_now.jpg')

    # Check if the blend_image file was uploaded
    if 'blend_image' not in request.files:
        return "No file part"

    blend_image = request.files['blend_image']

    # Check if a file was selected
    if blend_image.filename == '':
        return "No selected file"

    # Save the uploaded blend_image to the target folder
    blend_image.save(os.path.join(target_folder, 'blend.jpg'))

    # Open the source and blend images
    img1 = Image.open(source_image_path)
    img2 = Image.open(os.path.join(target_folder, 'blend.jpg'))

    # Resize images to match dimensions
    width = max(img1.size[0], img2.size[0])
    height = max(img1.size[1], img2.size[1])

    img1 = img1.resize((width, height), Image.ANTIALIAS)
    img2 = img2.resize((width, height), Image.ANTIALIAS)

    # If image1 is grayscale, convert image2 to grayscale
    if len(img1.mode) < 3:
        img2 = img2.convert('L')

    # Blend images using the specified alpha
    blended_img = Image.blend(img1, img2, float(alpha) / 100)

    # Save the blended image
    blended_img_path = os.path.join(target_folder, 'img_now.jpg')
    blended_img.save(blended_img_path)

    # Perform additional image processing (histogram_rgb) in a separate thread
    processing_thread = threading.Thread(
        target=Image_Processing.both_histogram(blended_img_path))
    processing_thread.start()
    processing_thread.join()

    return render_template("processing.html", image_name=image_name)


@app.route("/brightness_addition", methods=["POST"])
def brightness_addition():

    image_name = request.form['image']

    # Perform brightness addition
    Image_Processing.brightness_addition()

    # Perform additional image processing (histogram_rgb) in a separate thread
    processing_thread = threading.Thread(
        target=Image_Processing.both_histogram('static/images/img_now.jpg'))
    processing_thread.start()
    processing_thread.join()

    return render_template("processing.html", image_name=image_name)


@app.route("/brightness_subtraction", methods=["POST"])
def brightness_subtraction():
    image_name = request.form['image']
    # Perform brightness subtraction
    Image_Processing.brightness_substraction()
    # Perform additional image processing (histogram_rgb) in a separate thread
    processing_thread = threading.Thread(
        target=lambda: Image_Processing.both_histogram('static/images/img_now.jpg'))
    processing_thread.start()
    processing_thread.join()
    return render_template("processing.html", image_name=image_name)


@app.route("/brightness_multiplication", methods=["POST"])
def brightness_multiplication():
    image_name = request.form['image']
    # Perform brightness multiplication
    Image_Processing.brightness_multiplication()
    # Perform additional image processing (histogram_rgb) in a separate thread
    processing_thread = threading.Thread(
        target=lambda: Image_Processing.both_histogram('static/images/img_now.jpg'))
    processing_thread.start()
    processing_thread.join()
    return render_template("processing.html", image_name=image_name)


@app.route("/brightness_division", methods=["POST"])
def brightness_division():
    image_name = request.form['image']
    # Perform brightness division
    Image_Processing.brightness_division()
    # Perform additional image processing (histogram_rgb) in a separate thread
    processing_thread = threading.Thread(
        target=lambda: Image_Processing.both_histogram('static/images/img_now.jpg'))
    processing_thread.start()
    processing_thread.join()
    return render_template("processing.html", image_name=image_name)


@app.route("/thresholding", methods=["POST"])
def thresholding():
    image_name = request.form['image']

    lower_thres = int(request.form['lower_thres'])
    upper_thres = int(request.form['upper_thres'])
    Image_Processing.threshold(lower_thres, upper_thres)
    processing_thread = threading.Thread(
        target=lambda: Image_Processing.both_histogram('static/images/img_now.jpg'))
    processing_thread.start()
    processing_thread.join()

    return render_template("processing.html", image_name=image_name)


@app.route("/puzzle", methods=["POST"])
def puzzle():
    piece = int(request.form['piece'])

    processing_thread = threading.Thread(
        target=lambda: Image_Processing.puzzle(piece))
    processing_thread.start()
    processing_thread.join()
    image_urls = []
    for i in range(piece):
        row = []
        for j in range(piece):
            # Construct the image file path
            image_path = f"static/temp/puzzle/image_piece_{i}_{j}.jpg"
            row.append(image_path)
        image_urls.append(row)

    return render_template('puzlle.html', image_urls=image_urls, size=piece)


@app.route("/puzzle_random", methods=["POST"])
def puzzle_random():
    piece = int(request.form['piece'])

    processing_thread = threading.Thread(
        target=lambda: Image_Processing.puzzle(piece))
    processing_thread.start()
    processing_thread.join()
    image_urls = []
    for i in range(piece):
        row = []
        for j in range(piece):
            # Construct the image file path
            image_path = f"static/temp/puzzle/image_piece_{i}_{j}.jpg"
            row.append(image_path)
        image_urls.append(row)

    random.shuffle(image_urls)
    for row in image_urls:
        random.shuffle(row)
    return render_template('puzlle.html', image_urls=image_urls, size=piece)


@app.route("/IdentityFilter", methods=["POST"])
def IdentityFilter():
    filename = request.form['image']
    Image_Processing.identityFilter()
    processing_thread = threading.Thread(
        target=Image_Processing.both_histogram('static/images/img_now.jpg'))
    processing_thread.start()

    # Wait for the processing thread to finish
    processing_thread.join()
    return render_template("processing.html", image_name=filename)


@app.route("/MeanFilter", methods=["POST"])
def MeanFilter():
    filename = request.form['image']
    Image_Processing.MeanFilter()
    processing_thread = threading.Thread(
        target=Image_Processing.both_histogram('static/images/img_now.jpg'))
    processing_thread.start()

    # Wait for the processing thread to finish
    processing_thread.join()
    return render_template("processing.html", image_name=filename)


@app.route("/Scaled_Blur", methods=["POST"])
def Scaled_Blur():
    filename = request.form['image']
    factor = int(request.form['Blur'])
    Image_Processing.BetterBlur(factor)
    processing_thread = threading.Thread(
        target=Image_Processing.both_histogram('static/images/img_now.jpg'))
    processing_thread.start()

    return render_template("processing.html", image_name=filename)


@app.route("/Gaussian_Blur", methods=["POST"])
def Gaussian_Blur():
    filename = request.form['image']
    factor = int(request.form['Blur'])
    Image_Processing.GaussianBlur(factor)
    processing_thread = threading.Thread(
        target=Image_Processing.both_histogram('static/images/img_now.jpg'))
    processing_thread.start()

    return render_template("processing.html", image_name=filename)


@app.route("/Median_Blur", methods=["POST"])
def Median_Blur():
    filename = request.form['image']
    factor = int(request.form['Blur'])
    Image_Processing.MedianBlur(factor)
    processing_thread = threading.Thread(
        target=Image_Processing.both_histogram('static/images/img_now.jpg'))
    processing_thread.start()

    return render_template("processing.html", image_name=filename)


@app.route("/Billateral", methods=["POST"])
def Billateral():
    filename = request.form['image']
    Dimension = int(request.form['Dimension'])
    sigma_color = int(request.form['sigma_color'])
    sigma_space = int(request.form['sigma_space'])

    Image_Processing.BillateralFilter(Dimension, sigma_color, sigma_space)
    processing_thread = threading.Thread(
        target=Image_Processing.both_histogram('static/images/img_now.jpg'))
    processing_thread.start()

    return render_template("processing.html", image_name=filename)


@app.route("/Option_Filter", methods=["POST"])
def Option_Filter():
    filename = request.form['image']
    mode = request.form['filter_mode']

    Image_Processing.choose_filter(mode)
    processing_thread = threading.Thread(
        target=Image_Processing.both_histogram('static/images/img_now.jpg'))
    processing_thread.start()

    return render_template("processing.html", image_name=filename)


if __name__ == "__main__":
    app.run()
