import os
import json
import glob
import logging
import time
import requests
import base64

#Creating logging file and configuring it
LOG_FILE_PATH = '/tmp/sending_alert_data.log'
logging.basicConfig(filename=LOG_FILE_PATH, level=logging.INFO, format='%(asctime)s: %(levelname)s: %(message)s')
logging.info('CODE STARTED')

# Tarsyer server externalIP and Port
url = "http://dashboard.alt1.tarsyer.com:5013/acw_alert_server"
logging.info(f'URL : {url}')

# alert folders variable and if folders are not present create it
# Basically 3 folder :
# 1 - where cv code save images :: ALERT_FOLDER_PATH
# 2 - after sending the save image move it to deleted folder :: deleted_folder
# 3 - If due to some reason, alert didn't got sent move it to home dir folder :: ALERT_FOLDER_HOMEDIR
ALERT_FOLDER_PATH = '/tmp/alert_images/'
if not os.path.exists(ALERT_FOLDER_PATH):
    os.system(f'mkdir -p {ALERT_FOLDER_PATH}')
    logging.info(f'creating folder :{ALERT_FOLDER_PATH}')

ALERT_FOLDER_HOMEDIR = '/home/pi/alert_images/'
if not os.path.exists(ALERT_FOLDER_HOMEDIR):
    os.system(f'mkdir -p {ALERT_FOLDER_HOMEDIR}')
    logging.info(f'creating folder : {ALERT_FOLDER_HOMEDIR}')

deleted_folder = '/tmp/deleted_alert_images/'
if not os.path.exists(deleted_folder):
    os.system(f'mkdir -p {deleted_folder}')
    logging.info(f'creating folder : {deleted_folder}')

# defining the header i.e type of payload getting sent and bearer token
headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer Tarsyer_CV_2024'
}

"""
alert file name structure :: alert-datetime_location_111.jpg

alert-datetime - 2024-02-01_16-39-00 :: %Y-%m-%d_%H-%M-%S
location - WHRS-Load-Cell

111 - explained below :: 1 - present, 0 - absent
1st - helmet
2nd - suit
3rd - gloves
"""


# This is separate function to check and sent the files present in home dir :: ALERT_FOLDER_HOMEDIR
# It is seperate because logging is different and we don't need to move again and again to the home dir
# ALERT_FOLDER_HOMEDIR_files contains list of all the files present in the home dir

def sending_homedir_files(ALERT_FOLDER_HOMEDIR_files):

    # Iterate through each file in the ALERT_FOLDER_PATH.
    for alert_files in ALERT_FOLDER_HOMEDIR_files:

        # Extract information from the filename for datetime, location, and alert output.
        alert_files_name_list = (alert_files.split('/')[-1]).split('_')
        alert_datetime = alert_files_name_list[0] + "_" + alert_files_name_list[1]
        alert_location = alert_files_name_list[2]
        alert_output = alert_files_name_list[3].split(".")[0]

        # Read the image file in binary mode and convert it to a base64-encoded string.
        with open(alert_files, "rb") as img_file:
            alert_img_string = str(base64.b64encode(img_file.read()))

        # Create a dictionary containing various details of an alert, such as datetime, location, alert message, and image data.
        alert_string = {
            "date_time": alert_datetime,
            "location": alert_location,
            "alert": alert_output,
            "image": alert_img_string
        }

        # Convert the dictionary into a JSON-formatted string.
        payload = json.dumps(alert_string)

        # Try to send a POST request to the specified URL with the JSON payload and set a timeout of 10 seconds.
        try:
            response = requests.request("POST", url, headers=headers, data=payload, timeout=10)

            # Log the response text for debugging purposes.
            logging.info(response.text)

            if "success" in response.text:
                # Move the alert files to a 'deleted_folder' after successful transmission.
                os.system(f'mv {alert_files} {deleted_folder}')
                logging.info(f'INITIAL: alert file moved to {deleted_folder}')
            else:
                logging.info(f'`success` is not present in the response')

        # Handle a timeout exception that might occur during the request.
        except requests.Timeout:
            logging.info(f'INITIAL: ERROR ERROR :: Timeout')

        # Handle other request exceptions and log the error message.
        except requests.RequestException as error:
            logging.info(f'INITIAL: ERROR ERROR :: {error}')


while True:

    # Get a list of all .jpg files in the ALERT_FOLDER_HOMEDIR directory.
    ALERT_FOLDER_HOMEDIR_files = glob.glob(f'{ALERT_FOLDER_HOMEDIR}/*.jpg')

    # Check if the length of the list is not zero.
    if len(ALERT_FOLDER_HOMEDIR_files) != 0:
        logging.info(f'len of {ALERT_FOLDER_HOMEDIR_files} is not zero')

        # Call the sending_homedir_files function to handle the files in ALERT_FOLDER_HOMEDIR.
        sending_homedir_files(ALERT_FOLDER_HOMEDIR_files)

    # Get a list of all .jpg files in the ALERT_FOLDER_PATH directory.
    ALERT_FOLDER_PATH_files = glob.glob(f'{ALERT_FOLDER_PATH}/*.jpg')

    # Iterate through each file in the ALERT_FOLDER_PATH.
    for alert_files in ALERT_FOLDER_PATH_files:

        # Extract information from the filename for datetime, location, and alert output.
        alert_files_name_list = (alert_files.split('/')[-1]).split('_')
        alert_datetime = alert_files_name_list[0] + "_" + alert_files_name_list[1]
        alert_location = alert_files_name_list[2]
        alert_output = alert_files_name_list[3].split(".")[0]

        # Read the image file in binary mode and convert it to a base64-encoded string.
        with open(alert_files, "rb") as img_file:
            alert_img_string = str(base64.b64encode(img_file.read()))

        # Create a dictionary with alert details.
        alert_string = {
            "date_time": alert_datetime,
            "location": alert_location,
            "alert": alert_output,
            "image": alert_img_string
        }

        # Convert the dictionary into a JSON-formatted string.
        payload = json.dumps(alert_string)

        try:
            # Send a POST request with the payload to the specified URL.
            response = requests.request("POST", url, headers=headers, data=payload, timeout=10)
            logging.info(response.text)
            if "success" in response.text:
                # Move the processed alert file to the deleted_folder.
                os.system(f'mv {alert_files} {deleted_folder}')
                logging.info(f'alert file moved to {deleted_folder}')
            else:
                logging.info(f'`success` is not present in the response')

        # Handle a timeout exception during the request.
        except requests.Timeout:
            logging.info(f'ERROR ERROR :: Timeout')

            # Move the file to ALERT_FOLDER_HOMEDIR in case of a timeout.
            logging.info(f'file moved to {ALERT_FOLDER_HOMEDIR}')
            os.system(f'mv {alert_files} {ALERT_FOLDER_HOMEDIR}')

        # Handle other request exceptions and log the error message.
        except requests.RequestException as error:
            logging.info(f'ERROR ERROR :: {error}')

            # Move the file to ALERT_FOLDER_HOMEDIR in case of an exception.
            logging.info(f'file moved to {ALERT_FOLDER_HOMEDIR}')
            os.system(f'mv {alert_files} {ALERT_FOLDER_HOMEDIR}')

        # Introduce a delay of 1 second for resting between iterations.
        time.sleep(1)

    # Sleep for 60 seconds after processing all files.
    time.sleep(60)
