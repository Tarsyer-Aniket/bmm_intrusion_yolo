# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import time
import logging

class CentroidTracker():
	def __init__(self, camera_detail, line, entry_point, maxDisappeared=3, maxDistance=50, minDistance=20):
		
		self.nextObjectID = 1
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()
		self.startTime = OrderedDict()

		self.maxDisappeared = maxDisappeared
		self.maxDistance = maxDistance
		
		self.minDistance = minDistance
		self.startCentroid = OrderedDict()
		self.count=0

		self.line = line
		self.entry_point = entry_point
		print(self.line)
		print(self.entry_point)

		self.entry_area_polarity = self._point_side_of_line(self.line, self.entry_point)

		self.positive_direction_count = 0
		self.negative_direction_count = 0
		self.ignore_count = 0

		self.extended_positive_direction_count = 0
		self.extended_negative_direction_count = 0
		self.extended_ignore_count = 0

		self.vsq_logger = logging.getLogger('CT_Event')
		self.vsq_logger.setLevel(logging.INFO)
		formatter = logging.Formatter('%(asctime)s: %(levelname)s: %(message)s')

		VSQ_LOG_PATH = f'/tmp/centroid_{camera_detail}.log'

		self.log_handler = logging.FileHandler(VSQ_LOG_PATH, mode='a')
		self.log_handler.setFormatter(formatter)
		self.vsq_logger.addHandler(self.log_handler)

		self.vsq_logger.info(f'Centroid Started {camera_detail}')
		self.vsq_logger.info(f'Store is {self.entry_area_polarity} side [True means right and False means left]')
		self.vsq_logger.info(f'objectID \t startCentroid \t lastCentroid \t axis_dist \t dist. \t +ve dir \t -ve dir \t NA count \t Status')


	def _point_side_of_line(self, line, point):
		x1, y1 = line[0]
		x2, y2 = line[1]
		x, y = point
		# Vector AB
		AB_x = x2 - x1
		AB_y = y2 - y1

		# Vector AP
		AP_x = x - x1
		AP_y = y - y1

		# Calculate cross product
		cross_product = (AB_x * AP_y) - (AB_y * AP_x)
		# Interpret the result
		if cross_product < 0:
			return False
		elif cross_product >= 0:
			return True
		# else:
		# 	return "Store is on the line"


	# assumes line segments are stored in the format [(x0,y0),(x1,y1)]
	def _intersects(self, s0, s1):
		dx0 = s0[1][0]-s0[0][0]
		dx1 = s1[1][0]-s1[0][0]
		dy0 = s0[1][1]-s0[0][1]
		dy1 = s1[1][1]-s1[0][1]
		p0 = dy1*(s1[1][0]-s0[0][0]) - dx1*(s1[1][1]-s0[0][1])
		p1 = dy1*(s1[1][0]-s0[1][0]) - dx1*(s1[1][1]-s0[1][1])
		p2 = dy0*(s0[1][0]-s1[0][0]) - dx0*(s0[1][1]-s1[0][1])
		p3 = dy0*(s0[1][0]-s1[1][0]) - dx0*(s0[1][1]-s1[1][1])
		return (p0*p1<=0) & (p2*p3<=0)


	def _extended_intersects(self, line, line2):

		x1, y1 = line[0]
		x2, y2 = line[1]

		prev_pos = line2[0]
		curr_pos = line2[1]

		
		# Check if a person has crossed a line defined by two points (x1, y1) and (x2, y2) in a particular direction.
		
		# :param x1, y1: Coordinates of the first point of the line
		# :param x2, y2: Coordinates of the second point of the line
		# :param prev_pos: The previous position of the person as a tuple (x, y)
		# :param curr_pos: The current position of the person as a tuple (x, y)
		# :return: 'crossed' if the line was crossed, 'same_side' if still on the same side, or 'no_cross' if on the opposite side without crossing
		

		def position_to_line(px, py):
			# Returns a tuple with the determinant and its sign
			det = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
			return det, np.sign(det)

		prev_det, prev_sign = position_to_line(*prev_pos)
		curr_det, curr_sign = position_to_line(*curr_pos)

		# Check if the line was crossed by looking at the sign change
		if prev_sign != curr_sign:
			# return 'crossed'
			return True
		elif prev_sign == curr_sign and curr_det == 0:
			# The person is exactly on the line at the current position
			# return 'no_cross'
			return True
		else:
			# The person is still on the same side of the line
			# return 'same_side'
			return False

	def register(self, centroid):
		
		self.objects[self.nextObjectID] = centroid
		self.startCentroid[self.nextObjectID] = centroid
		self.count+=1
		self.disappeared[self.nextObjectID] = 0
		self.startTime[self.nextObjectID] = int(time.time())
		self.nextObjectID += 1

	def deregister(self, objectID):

		# axis_dist = int(self.objects[objectID][self.index] - self.startCentroid[objectID][self.index])

		end_point = self.objects[objectID]
		start_point = self.startCentroid[objectID]
		displacement_path_line = (start_point, end_point)

		object_polarity = self._point_side_of_line(self.line, end_point)

		crossed = self._intersects(self.line, displacement_path_line)

		if crossed:
			if object_polarity == self.entry_area_polarity:
				self.positive_direction_count+=1
				string = 'POS +1'
			else:
				self.negative_direction_count+=1
				string = 'NEG +1'
		else:
			self.ignore_count+=1
			string = 'IGN +1'

		extended_crossed = self._extended_intersects(self.line, displacement_path_line)

		if extended_crossed:
			if object_polarity == self.entry_area_polarity:
				self.extended_positive_direction_count+=1
				extended_string = 'EXT POS +1'
			else:
				self.extended_negative_direction_count+=1
				extended_string = 'EXT NEG +1'
		else:
			self.extended_ignore_count+=1
			extended_string = 'EXT IGN +1'


		dist_ = round(np.linalg.norm(np.array(self.startCentroid[objectID])-np.array(self.objects[objectID])),2)
		#print('distance = {}'.format(dist_))
		if dist_<self.minDistance:
			self.count-=1

		self.vsq_logger.info(f'{objectID} \t! {self.startCentroid[objectID]} \t! {self.objects[objectID]} \t! {string} \t! {extended_string} ')
		# self.tracker_logging.info(f'{objectID} \t {self.startCentroid[objectID]} \t {self.objects[objectID]} \t {axis_dist} \t {dist_} \t {self.positive_direction_count} \t {self.negative_direction_count} \t {self.ignore_count} \t {status}')
		# self.tracker_logging.info(f'Object ID : {objectID}, Object_start_point : {self.startCentroid[objectID]}, object_end_point: {np.array(self.objects[objectID])}, distance : {dist_}, status : {status}')
		# self.tracker_logging.info(f'Object ID : {objectID}, +ve count : {self.positive_direction_count}, -ve count : {self.negative_direction_count}, ignore count : {self.ignore_count}')

		
		del self.objects[objectID]
		del self.disappeared[objectID]
		del self.startTime[objectID]
		del self.startCentroid[objectID]

	def update(self, rects):
		
		if len(rects) == 0:
			try:
				for objectID in self.disappeared.keys():
					self.disappeared[objectID] += 1

					
					if self.disappeared[objectID] > self.maxDisappeared:
						self.deregister(objectID)
			except Exception as e:
				self.vsq_logger.info(f'Exception ******* {e}')
				print('..................***********************')

			return self.objects, self.startTime

		inputCentroids = np.zeros((len(rects), 2), dtype="int")

		for (i, (startX, startY, endX, endY)) in enumerate(rects):
			cX = int((startX + endX) / 2.0)
			cY = int((startY + endY) / 2.0)
			inputCentroids[i] = (cX, cY)

		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids)):
				self.register(inputCentroids[i])

		else:
			objectIDs = list(self.objects.keys())
			objectCentroids = list(self.objects.values())

			D = dist.cdist(np.array(objectCentroids), inputCentroids)

			rows = D.min(axis=1).argsort()

			cols = D.argmin(axis=1)[rows]

			usedRows = set()
			usedCols = set()

			for (row, col) in zip(rows, cols):
				if row in usedRows or col in usedCols:
					continue


				if D[row, col] > self.maxDistance:
						self.register(inputCentroids[col])
						usedRows.add(row)
						usedCols.add(col)
						continue

				objectID = objectIDs[row]
				self.objects[objectID] = inputCentroids[col]
				self.disappeared[objectID] = 0

				usedRows.add(row)
				usedCols.add(col)

			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)

			
			for row in unusedRows:
				objectID = objectIDs[row]
				self.disappeared[objectID] += 1

				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)

			for col in unusedCols:
				self.register(inputCentroids[col])

		return self.objects, self.startTime
		#return self.objects
