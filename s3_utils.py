import os
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

# Function to check AWS environment variables
def check_env_variables():
    """
    Checks if AWS credentials and region are set in environment variables.
    
    Raises:
        EnvironmentError: If any of the required AWS environment variables are missing.
    """
    required_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_REGION']
    missing_vars = [var for var in required_vars if var not in os.environ]

    if missing_vars:
        raise EnvironmentError(f"Missing AWS environment variables: {', '.join(missing_vars)}")
    print("All required AWS environment variables are set.")

# Function to upload one or many files to S3
def upload(paths, bucket_name, s3_prefix=""):
    """
    Uploads a file or a list of files to an S3 bucket.

    Args:
        paths (str or list of str): Path or list of paths to the files to upload.
        bucket_name (str): Name of the S3 bucket.
        s3_prefix (str): Optional S3 prefix (folder) where the files will be uploaded.

    Returns:
        None
    """
    s3_client = boto3.client('s3')

    # Ensure paths is a list
    if isinstance(paths, str):
        paths = [paths]

    for file_path in paths:
        if os.path.isfile(file_path):
            s3_key = os.path.join(s3_prefix, os.path.basename(file_path))
            try:
                s3_client.upload_file(file_path, bucket_name, s3_key)
                print(f"Uploaded {file_path} to s3://{bucket_name}/{s3_key}")
            except Exception as e:
                print(f"Failed to upload {file_path}. Error: {e}")
        else:
            print(f"{file_path} is not a valid file.")

# Function to download one or many files from S3
def download(paths, bucket_name, local_dir="."):
    """
    Downloads a file or a list of files from an S3 bucket.

    Args:
        paths (str or list of str): S3 key or list of S3 keys to download.
        bucket_name (str): Name of the S3 bucket.
        local_dir (str): Local directory where files will be downloaded.

    Returns:
        None
    """
    s3_client = boto3.client('s3')

    # Ensure paths is a list
    if isinstance(paths, str):
        paths = [paths]

    for s3_key in paths:
        local_file_path = os.path.join(local_dir, os.path.basename(s3_key))
        try:
            s3_client.download_file(bucket_name, s3_key, local_file_path)
            print(f"Downloaded s3://{bucket_name}/{s3_key} to {local_file_path}")
        except Exception as e:
            print(f"Failed to download {s3_key}. Error: {e}")