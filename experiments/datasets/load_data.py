from os.path import isfile
import logging
import requests
import pickle
import sys
# Setting up the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
stdout_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stdout_handler)
def get_data(file_name, local=False):
    '''
    Fetches the file from the location, loads it into memory and returns the data.
    '''
    if local:
        logger.info(f"Opening {file_name}...")
        with open(file_name, 'rb') as handle:
            data = pickle.load(handle)
        return data
    else:
        if not isfile(file_name):
            url = f"https://ussf-ssac-23-soccer-gnn.s3.us-east-2.amazonaws.com/public/counterattack/{file_name}"

            logger.info(f"Downloading data from {url}...")

            r = requests.get(url, stream=True)
            # Print progress bar
            block_size = 1024
            n_chunk = 1000
            with open(file_name, 'wb') as f:
                for i, chunk in enumerate(r.iter_content(block_size)):
                    f.write(chunk)
                    if i % n_chunk == 0:
                        logger.info(f"Downloaded {i//n_chunk}%")
            logger.info("File downloaded successfully!")

        logger.info(f"Opening {file_name}...")
        with open(file_name, 'rb') as handle:
            data = pickle.load(handle)
        return data