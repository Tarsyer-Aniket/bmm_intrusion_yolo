import smtplib,ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os
import random
import time
import logging

email_logger = logging.getLogger('email_alerts')
email_logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s: %(levelname)s: %(message)s')

CL_LOG_PATH = f'/tmp/send_mail.log'

log_handler = logging.FileHandler(CL_LOG_PATH, mode='a')
log_handler.setFormatter(formatter)
email_logger.addHandler(log_handler)
email_logger.info("Code Started")


def send_mail(sender_email,password, mail_message, img_path, LOGGER):

    alert_log = LOGGER

    # creates SMTP session 
    smtp_server = "smtp.gmail.com"
    port = 587  # For starttls

    # Create a secure SSL context
    context = ssl.create_default_context()

    # Try to log in to server and send email
    try:
        server = smtplib.SMTP(smtp_server,port)
        server.ehlo() # Can be omitted
        server.starttls(context=context) # Secure the connection
        server.ehlo() # Can be omitted
        server.login(sender_email, password)
        receiver_email_id=["aniket.tayade@tarsyer.com"]
        #cc_email_id=["aniket.tayade@tarsyer.com"]
        cc_email_id=["ashutosh.bhagwat@tarsyer.com", "aniket.tayade@tarsyer.com", "vnayar@tarsyer.com"]


        # Setting up the mail
        message = MIMEMultipart("alternative")
        message["Subject"] = "Warning! - Intrusion detected in RMHD Apron Feeder area"
        message["From"] = sender_email
        message["To"] = ", ".join(receiver_email_id)
        message["Cc"] = ", ".join(cc_email_id)

        # Attaching text
        text = MIMEText(mail_message, "html")
        text.add_header('Content-Type', 'text/html')
        message.attach(text)

        # New way of attaching Image
        f = open(img_path, 'rb')
        # set attachment mime and file name, the image type is png
        img_name = f"{img_path.split('/')[-1]}"
        mime = MIMEBase('image', 'jpg', filename=img_name)
        # add required header data:
        mime.add_header('Content-Disposition', 'attachment', filename=img_name)
        mime.add_header('X-Attachment-Id', '0')
        mime.add_header('Content-ID', '<0>')
        # read attachment file content into the MIMEBase object
        mime.set_payload(f.read())
        # encode with base64
        encoders.encode_base64(mime)
        # add MIMEBase object to MIMEMultipart object
        message.attach(mime)

        # Old way of attaching an image
        # f = open(img_path, 'rb')
        # img = f.read()
        # image_data = MIMEImage(img, name="Alert Image")
        # message.attach(image_data)
        composed_msg = message.as_string()

        # sending the mail 
        alert_log.info(f'receiver : {receiver_email_id+cc_email_id}, msg : {mail_message}')
        server.sendmail(sender_email, receiver_email_id+cc_email_id, composed_msg)

        return True


    except Exception as e:
        # Print any error messages to stdout
        alert_log.info(str(e))
        print(e)
        return False
    finally:
        server.quit() 
        s = smtplib.SMTP('smtp.gmail.com', 587) 

        return False
        
if __name__ == '__main__':
    sender_email_id="tarsyer.imp.alerts@gmail.com"
    sender_email_id_password="ljtvzmqbgtpodmvn"
    name = 'TEST AREA'

    formatted_date_string = '1970/1/1 12:20:00'
    # msg = "<p>Hey there ,<br>This is a test mail</br></p><p>Regards, <br>Tarsyer Insights</br></p>"
    msg = f'<p>Hello,</p> <p>This is to bring to your notice that a intrusion has detected in the {name} area.</p><br> <p>Date & Time : {formatted_date_string} </p> <br> <p>Please take immediate action to resolve the issue and prevent any incident. An image of the incident is attached, for your review.</p> <p>In case the situation is not resolved, another alert will be sent after 5 mins.</p> <p>Thank you</p> <p>Team Tarsyer</p> <p><a href="http://www.tarsyer.com" target="_blank">www.tarsyer.com</a></p>'
    img_path = r'Tarsyer_Logo_BrandName.png'
    send_mail(sender_email_id, sender_email_id_password, msg, img_path, LOGGER=email_logger)
