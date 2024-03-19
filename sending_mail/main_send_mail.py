import os
import json
import glob
import logging
from send_mail_tarsyer import email_logger, send_mail
import time
from datetime import datetime

"""
alert file name structure :: datetime_location_111.jpg

alert-datetime - 2024-02-01_16-39-00 :: %Y-%m-%d_%H-%M-%S
location - WHRS-Load-Cell

111 - explained below :: 1 - present, 0 - absent
1st - helmet
2nd - suit
3rd - gloves
"""


sender_email_id="tarsyer.imp.alerts@gmail.com"
sender_email_id_password="ljtvzmqbgtpodmvn"
mobile_no = '7057530791'
# msg = "<p>Hey there ,<br>This is a test mail</br></p><p>Regards, <br>Tarsyer Insights</br></p>"
# img_path = r'Tarsyer_Logo_BrandName.png'
name = "RMHD Apron Feeder"
ALERT_FOLDER_PATH = '/tmp/alert_images_for_email/'
if not os.path.exists(ALERT_FOLDER_PATH):
    os.system(f'mkdir -p {ALERT_FOLDER_PATH}')
    email_logger.info(f'creating folder :{ALERT_FOLDER_PATH}')

ALERT_FOLDER_HOMEDIR = '/home/pi/alert_images_for_email/'
if not os.path.exists(ALERT_FOLDER_HOMEDIR):
    os.system(f'mkdir -p {ALERT_FOLDER_HOMEDIR}')
    email_logger.info(f'creating folder : {ALERT_FOLDER_HOMEDIR}')

deleted_folder = '/tmp/deleted_alert_images_for_email/'
if not os.path.exists(deleted_folder):
    os.system(f'mkdir -p {deleted_folder}')
    email_logger.info(f'creating folder : {deleted_folder}')



while True:

    # Get a list of all .jpg files in the ALERT_FOLDER_PATH directory.
    ALERT_FOLDER_PATH_files = glob.glob(f'{ALERT_FOLDER_PATH}*.jpg')

    # Iterate through each file in the ALERT_FOLDER_PATH.
    for alert_files in sorted(ALERT_FOLDER_PATH_files):
        # Extract information from the filename for datetime, location, and alert output.
        alert_files_name_list = (alert_files.split('/')[-1]).split('_')
        alert_datetime = alert_files_name_list[0] + "_" + alert_files_name_list[1]
        alert_location = alert_files_name_list[2]
        alert_output = alert_files_name_list[3].split(".")[0]

        original_date = datetime.strptime(alert_datetime, "%Y-%m-%d_%H-%M-%S")
        formatted_date_string = original_date.strftime("%d-%m-%Y %H:%M:%S")        

        img_path = alert_files

        # msg = f"<p> <b> Safety Violation at WHRS </b> </p><p> <br> Date & Time : {formatted_date_string}</br> <br>{alert_details_1}</br><br>{alert_details_2}</br><br>{alert_details_3}</br></p><p>Regards, <br>Tarsyer Insights</br></p>"

        msg = f'<p>Hello,</p> <p>This is to bring to your notice that a intrusion has detected in the {name} area.</p><br> <p>Date & Time : {formatted_date_string} </p> <br> <p>Please take immediate action to resolve the issue and prevent any incident. An image of the incident is attached, for your review.</p> <p>In case the situation is not resolved, another alert will be sent after 5 mins.</p> <p>Thank you</p> <p>Team Tarsyer</p> <p><a href="http://www.tarsyer.com" target="_blank">www.tarsyer.com</a></p>'

        status = send_mail(sender_email_id, sender_email_id_password, msg, img_path, LOGGER=email_logger)

        if status:
            os.system(f'mv {alert_files} {deleted_folder}')
            email_logger.info(f'alert file moved to {deleted_folder}')
        else:
            os.system(f'mv {alert_files} {ALERT_FOLDER_HOMEDIR}')
            email_logger.info(f'file moved to {ALERT_FOLDER_HOMEDIR}')

        time.sleep(1)

    time.sleep(1)
