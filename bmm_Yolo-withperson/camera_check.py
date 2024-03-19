import datetime
import pathlib
import time
import os

time.sleep(10)
PATH = '/tmp/camera.txt'
computer_vision_service_file_name = 'tlm_bag_counting.service'

while True:

	if not os.path.exists(PATH):
		os.system(f'systemctl --user restart {computer_vision_service_file_name}')
		time.sleep(10)

	filename = pathlib.Path(f'{PATH}')
	modify_timestamp = filename.stat().st_mtime
	#print(modify_timestamp)
	modify_date = datetime.datetime.fromtimestamp(modify_timestamp)
	#print('Modified on:', modify_date)

	difference = datetime.datetime.now()-modify_date
	#print(difference.seconds)

	if difference.seconds > 10:
		print('time diff. > 10, so restart bag counting service file')
		os.system('systemctl --user restart tlm_bag_counting.service')

	time.sleep(10)