import cv2
from threading import Thread
import numpy as np
from queue import Queue
import time
import logging
from config import INPUT_SHAPE, PADDING

class VideoStreamQueue:
	def __init__(self, VIDEO_INPUT, CAMERA_DETAIL, CROPPING_COORD_PRI, SKIP_FRAME=5, queue_size=16):

		self.VIDEO_INPUT = VIDEO_INPUT
		self.CROPPING_COORD_PRI = CROPPING_COORD_PRI
		self.SKIP_FRAME = SKIP_FRAME
		self.stream = cv2.VideoCapture(self.VIDEO_INPUT)
		self.stopped = False

		self.Q = Queue(maxsize=queue_size)

		self.thread = Thread(target=self.update, args=())
		self.thread.daemon = True

		self.vsq_logger = logging.getLogger('VSQ_Event')
		self.vsq_logger.setLevel(logging.INFO)
		formatter = logging.Formatter('%(asctime)s: %(levelname)s: %(message)s')

		VSQ_LOG_PATH = f'/tmp/processing_{CAMERA_DETAIL}.log'

		self.log_handler = logging.FileHandler(VSQ_LOG_PATH, mode='a')
		self.log_handler.setFormatter(formatter)
		self.vsq_logger.addHandler(self.log_handler)


	def start(self):
		self.thread.start()
		return self

	def update(self):

		counter = 1

		while True:
			camera_image_dict = {}

			if self.stopped:
				break

			if not self.Q.full():
				
				if counter < self.SKIP_FRAME:
					grabbed = self.stream.grab()
					counter += 1
				else:
					
					grabbed, frame = self.stream.read()
					counter = 1 # reset the counter

					camera_image_dict['camera_status'] = grabbed


					if grabbed and frame is not(None):
						
						x1, y1, x2, y2 = self.CROPPING_COORD_PRI
						img = frame.copy()

						if PADDING:
							pass ###
						else:
							crop_img = img[y1:y2, x1:x2]
							image_300 = cv2.resize(crop_img, tuple(INPUT_SHAPE))

						camera_image_dict['image'] = image_300, frame

					else:
						camera_image_dict['image'] = None, frame
						self.stream.release()
						self.stream = cv2.VideoCapture(self.VIDEO_INPUT)
						continue

					
					self.Q.put(camera_image_dict)

			else:
				time.sleep(0.1)
				pass
			
			
				
			# if counter < self.SKIP_FRAME:
			# 	grabbed = self.stream.grab()
			# 	counter += 1
			# 	time.sleep(0.002)
			# else:
			# 	if not self.Q.full():

			# 		grabbed, frame = self.stream.read()
			# 		counter = 1 # reset the counter

			# 		camera_image_dict['camera_status'] = grabbed


			# 		if grabbed and frame is not(None):
						
			# 			height, width, _ = frame.shape
			# 			image_300 = cv2.resize(frame, tuple(INPUT_SHAPE))
			# 			camera_image_dict['image'] = image_300, frame

			# 		else:
			# 			camera_image_dict['image'] = None, frame
			# 			self.stream.release()
			# 			self.stream = cv2.VideoCapture(self.VIDEO_INPUT)
			# 			continue

					
			# 		self.Q.put(camera_image_dict)

			# 	else:
			# 		time.sleep(0.1)
			# 		pass

		self.stream.release()
		#result.release()

	def read(self):
		return self.Q.get()

	def running(self):
		return self.more() or not self.stopped

	def more(self):
		# return True if there are still frames in the queue. If stream is not stopped, try to wait a moment
		tries = 0
		while self.Q.qsize() == 0 and not self.stopped and tries < 5:
			time.sleep(0.1)
			tries += 1

		return self.Q.qsize() > 0

	def stop(self):
		self.stopped = True
		self.thread.join()
