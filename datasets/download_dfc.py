import requests
import zipfile
import io
import os
import sys

# URLs from https://ieee-dataport.org/competitions/2020-ieee-grss-data-fusion-contest

DFC_validation_URL = "https://ieee-dataport.s3.amazonaws.com/competition/17534/dfc_validation.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJOHYI4KJCE6Q7MIQ%2F20231031%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20231031T175102Z&X-Amz-SignedHeaders=Host&X-Amz-Expires=86400&X-Amz-Signature=253f2de95dfb40e654d718a4c65c319dd54d7322b53c4039f5369e6fde9e238b"
DFC_0_URL = "https://ieee-dataport.s3.amazonaws.com/competition/17534/dfc_0.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJOHYI4KJCE6Q7MIQ%2F20231031%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20231031T175102Z&X-Amz-SignedHeaders=Host&X-Amz-Expires=86400&X-Amz-Signature=b91bd9e9dac2e70bbddffc34fc608133608c9e6d2b419162bac9dbfda4469b4b"
S2_0_URL = "https://ieee-dataport.s3.amazonaws.com/competition/17534/s2_0.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJOHYI4KJCE6Q7MIQ%2F20231031%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20231031T175102Z&X-Amz-SignedHeaders=Host&X-Amz-Expires=86400&X-Amz-Signature=458ce6f622dbe666475ace26dac8d1d0f2aba5aa57daf39dc15d3b34a5b4d4a9"
S2_validation_URL = "https://ieee-dataport.s3.amazonaws.com/competition/17534/s2_validation.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJOHYI4KJCE6Q7MIQ%2F20231031%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20231031T175102Z&X-Amz-SignedHeaders=Host&X-Amz-Expires=86400&X-Amz-Signature=1746a71eab0788b245b51ad1d2c1774815d9ae0019117699602384b11dfce56e"


def download_and_unzip(url, output_directory="."):
    # Send a GET request to the URL
    response = requests.get(url)

    os.makedirs(output_directory, exist_ok=True)

    # Check if the request was successful
    if response.status_code == 200:
        # Open the zip file from the response content
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            # Extract all contents to the specified output directory
            zip_ref.extractall(output_directory)
        print("Download and extraction successful.")
    else:
        print("Failed to download the file.")


# Example usage
if __name__ == "__main__":

    script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))

    # Call the function to download and unzip the file
    print("downloading dfc_validation")
    download_and_unzip(DFC_validation_URL, os.path.join(script_directory, "dfc_validation"))

    print("downloading dfc_0")
    download_and_unzip(DFC_0_URL, os.path.join(script_directory, "dfc_0"))

    print("downloading s2_validation")
    download_and_unzip(S2_validation_URL, os.path.join(script_directory, "s2_validation"))

    print("downloading s2_0")
    download_and_unzip(S2_0_URL, os.path.join(script_directory, "s2_0"))