import gdown
import sys
import os

def download_file_from_drive(file_url, output_path):
    gdown.download(file_url, output_path, quiet=False)

script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))

# Example usage
file_url = "https://drive.google.com/uc?id=1Fxy-7fih22uTvMcHaqbGA8RjKzO6SDk6"
output_path = "RN18-epochepoch=123-val_lossval_loss=0.23.ckpt"
download_file_from_drive(file_url, os.path.join(script_directory, output_path))