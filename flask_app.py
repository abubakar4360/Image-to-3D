import os
import shutil
import subprocess
from flask import Flask, request, render_template, jsonify, send_file
import zipfile
from Segmentation import segmenatation

app = Flask(__name__)

def create_and_clean_directory(directory):
    if os.path.exists(directory):
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    else:
        os.makedirs(directory)

def create_zip():
    # Zip filename
    zip_filename = "output_files.zip"

    # Output directory where the files are located
    output_temp_dir = "temp_output"

    # Check if any directory exists in "output_temp_dir"
    directories = [d for d in os.listdir(output_temp_dir) if os.path.isdir(os.path.join(output_temp_dir, d))]

    if directories:
        directory_to_zip = os.path.join(output_temp_dir, directories[0])
        files_to_zip = [f for f in os.listdir(directory_to_zip) if f.endswith(".obj")]
    else:
        directory_to_zip = output_temp_dir
        files_to_zip = ["temp.obj", "temp.mtl", "temp_albedo.png"]

    # Create a ZipFile
    with zipfile.ZipFile(zip_filename, 'w') as zip_file:
        for file in files_to_zip:
            file_path = os.path.join(directory_to_zip, file)
            zip_file.write(file_path, os.path.basename(file_path))


    return zip_filename

def process_image(input_temp_dir):
    # Clean temp input directory
    create_and_clean_directory(input_temp_dir)

    if 'image' not in request.files:
        return jsonify({'error': 'Image not received'})

    file = request.files['image']

    allowed_extensions = {'png', 'jpg', 'jpeg'}
    if not file.filename.lower().endswith(tuple(allowed_extensions)):
        return jsonify({'error': 'Invalid file extension'})

    input_file_path = os.path.join(input_temp_dir, file.filename)
    file.save(input_file_path)

    return input_file_path

def run_subprocess(command):
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError:
        raise ValueError(f'Error in subprocess command: {command}')


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', success=False)

@app.route('/pymesh_dream', methods=['POST'])
def dream_gaussian_pymesh():
    os.system("pip install -U pymeshlab")
    return "Dependencies installed!"

@app.route('/pymesh_stable', methods=['POST'])
def stable_dreamfusion_pymesh():
    os.system("pip install -U pymeshlab==2022.2.post4")
    return "Dependencies installed!"

@app.route('/dream', methods=['GET','POST'])
def dream_gaussian():
    # Create a temporary directory for input
    input_temp_dir = 'dreamgaussian/temp'
    output_dir_path = 'temp_output'

    # Clear memory cache
    os.system('sync && echo 3 | sudo tee /proc/sys/vm/drop_caches')

    # Remove temp_output direcory from main
    if os.path.exists(output_dir_path):
        shutil.rmtree(output_dir_path)

    # Saving image
    input_file_path = process_image(input_temp_dir)

    # Applying segmentation on image
    image = segmenatation(input_file_path, input_temp_dir)

    # Run subprocess commands (this will give rgba image)
    command_stage0 = f"python dreamgaussian/process.py {image} --size 512"
    run_subprocess(command_stage0)

    # Read input file (rgba image)
    for file_rgba in os.listdir(input_temp_dir):
        if file_rgba.endswith("_rgba.png"):
            input_file = os.path.join(input_temp_dir, file_rgba)

    # Processing image to 3d (2 stage pipeline)
    command_stage1 = f"python dreamgaussian/main.py --config dreamgaussian/configs/image.yaml input={input_file} save_path=temp elevation=0 force_cuda_rast=True"
    command_stage2 = f"python dreamgaussian/main2.py --config dreamgaussian/configs/image.yaml input={input_file} save_path=temp elevation=0 force_cuda_rast=True"

    run_subprocess(command_stage1)
    run_subprocess(command_stage2)

    return render_template('index.html', success=True)


@app.route('/stable', methods=['GET','POST'])
def stable_dreamfusion():
    input_temp_dir = 'temp'
    output_dir_path = 'temp_output'

    # Clear memory cache
    os.system('sync && echo 3 | sudo tee /proc/sys/vm/drop_caches')

    # Remove temp_output direcory from main
    if os.path.exists(output_dir_path):
        shutil.rmtree(output_dir_path)

    # Store the original working directory
    original_directory = os.getcwd()

    try:
        os.chdir('stabledreamfusion')
        input_file_path = process_image(input_temp_dir)

        # Creating temp_output directory
        create_and_clean_directory(output_dir_path)

        # Run subprocess commands (this returns rgba image)
        command_stage0 = f"python preprocess_image.py {input_file_path}"
        run_subprocess(command_stage0)

        # Read input file
        for file_rgba in os.listdir(input_temp_dir):
            if file_rgba.endswith("_rgba.png"):
                input_file = os.path.join(input_temp_dir, file_rgba)

        # Processing image to 3D
        command_stage1 = f"python main.py -O --image {input_file} --workspace {output_dir_path} --iters 5000 --save_mesh --batch_size 2"
        run_subprocess(command_stage1)

        # Move 'temp_output' to the 'perfect365' directory
        shutil.move('temp_output', os.path.join(original_directory, 'temp_output'))

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        os.chdir(original_directory)  # Change back to the original directory

    return render_template('index.html', success=True)


@app.route('/one', methods=['GET','POST'])
def one2345():
    input_temp_dir = 'temp'
    output_dir_path = 'temp_output'

    # Clear memory cache
    # os.system('sync && echo 3 | sudo tee /proc/sys/vm/drop_caches')

    # Remove temp_output directory from main
    if os.path.exists(output_dir_path):
        shutil.rmtree(output_dir_path)

    # Store the original working directory
    original_directory = os.getcwd()

    try:
        os.chdir('One-2-3-45')
        create_and_clean_directory(input_temp_dir)

        input_file_path = process_image(input_temp_dir)
        # Applying segmentation on image
        image = segmenatation(input_file_path, input_temp_dir)

        # Read input file
        for file in os.listdir(input_temp_dir):
                input_file = os.path.join(input_temp_dir, file)

        # Processing image to 3D
        command_stage1 = f"python run.py --img_path {input_file} --half_precision"
        run_subprocess(command_stage1)

        # Move 'temp_output' to the 'perfect365' directory
        shutil.move('temp_output', os.path.join(original_directory, 'temp_output'))

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        os.chdir(original_directory)  # Change back to the original directory

    return render_template('index.html', success=True)


@app.route('/result', methods=['GET'])
def result():
    zipfile = create_zip()

    # Return zip file as response
    return send_file(zipfile, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5003)
