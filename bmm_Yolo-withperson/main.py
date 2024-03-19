import cv2
import numpy as np
import psutil
from config import *

from tarsyer import video_stream_queue as VSQ
from tarsyer import centroid_tracker as CT
from tarsyer import SSD_TfLite_Detector as OD

import time

import subprocess as sp
import shlex

# Adding logo
LOGO = cv2.imread('Tarsyer_Logo_BrandName.png', -1)
print(LOGO.shape)
LOGO = cv2.resize(LOGO, (100, 100))
y1, y2 = 0, 100
x1, x2 = 0, 100
alpha_s = LOGO[:, :, 3]/255.0
alpha_l = 1.0-alpha_s


camera_vsq = VSQ.VideoStreamQueue(VIDEO_INPUT, CAMERA_DETAIL, CROPPING_COORD_PRI, SKIP_FRAME, CAMERA_NO)
# video_cap = cv2.VideoCapture(0)
# video_cap.set(cv2.CAP_PROP_FPS, 10)
# CAMERA_FPS = video_cap.get(cv2.CAP_PROP_FPS)
# CAMERA_FPS = 30 (expected for webcam)
camera_vsq.vsq_logger.info('VSQ in progress')
camera_vsq.start()
total_frame_count = camera_vsq.stream.get(cv2.CAP_PROP_FRAME_COUNT)
print('total_frame_count = {}'.format(total_frame_count))


centroid_tracker = CT.CentroidTracker(CAMERA_DETAIL, LINE, ENTRY_POINT)

object_detector = OD.SSD_TfLite_Detection(CONFIDENCE_THRESH, MODEL_PATH, CLASSES=pretrained_class_names)

DEBUG = True
if DEBUG:
    startTime = time.monotonic()
    counter = 0
    fps = 0

if VIDEO_WRITE:
    # Open ffmpeg application as sub-process
    # FFmpeg input PIPE: RAW images in BGR color format
    # FFmpeg output MP4 file encoded with HEVC codec.
    # Arguments list:
    # -y                   Overwrite output file without asking
    # -s {width}x{height}  Input resolution width x height (1344x756)
    # -pixel_format bgr24  Input frame color format is BGR with 8 bits per color component
    # -f rawvideo          Input format: raw video
    # -r {fps}             Frame rate: fps (25fps)
    # -i pipe:             ffmpeg input is a PIPE
    # -vcodec libx265      Video codec: H.265 (HEVC)
    # -pix_fmt yuv420p     Output video color space YUV420 (saving space compared to YUV444)
    # -crf 24              Constant quality encoding (lower value for higher quality and larger output file).
    # {output_filename}    Output file name: output_filename (output.mp4)
    process = sp.Popen(shlex.split(f'/usr/bin/ffmpeg -y -s {VIDEO_WRITE_WIDTH}x{VIDEO_WRITE_HEIGHT} -pixel_format bgr24 -f rawvideo -r {saving_fps} -i pipe: -vcodec libx264 -pix_fmt yuv420p -crf 24 {output_filename}'), stdin=sp.PIPE)



### read counting file for checking initial count

###

camera_vsq.vsq_logger.info('CODE STARTED')

bag_count = 0
frame_counter = 0
person_counter = 0

first_check = True

purple_color = (100, 50, 100)  # (B, G, R)

while True:
    camera_dict = camera_vsq.read()
    camera_status = camera_dict['camera_status']
    # ret, frame = video_cap.read()

    if camera_status:
    # if ret:
        resized_frame, big_frame = camera_dict['image']

        # print(resized_frame.shape)

        # print(big_frame.shape)

        frame_counter += 1
        if frame_counter >= (total_frame_count/SKIP_FRAME) and 'rtsp' not in VIDEO_INPUT:
            break

        if DEBUG:
            counter += 1
            current_time = time.monotonic()
            
            if (current_time - startTime) > 10 or first_check:
                fps = counter / (current_time-startTime)
                counter = 0
                startTime = current_time
                # print('FPS = {}'.format(round(fps, 2)))
                camera_vsq.vsq_logger.info(f'FPS = {round(fps, 2)}')
                camera_vsq.vsq_logger.info(f'SKIP FRAME {camera_vsq.SKIP_FRAME}')
                load_average = os.getloadavg()
                ram = psutil.virtual_memory().percent
                swap_memory = psutil.swap_memory().percent
                try:
                    cpu = int(psutil.sensors_temperatures()['cpu_thermal'][0].current)
                except:
                    cpu = 'NA'
                first_check = False

            top_insights_img = np.zeros((100, 300, 3), dtype=np.uint8)
            top_insights_img[:, :] = purple_color

            cv2.putText(top_insights_img, f'FPS {round(fps, 1)} SF {camera_vsq.SKIP_FRAME} RAM {ram} SM {swap_memory}',
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(top_insights_img, f'CPU {cpu} LA {load_average}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 1)

        bbox_lists, class_lists, scores = object_detector.inference(resized_frame)

        imp_bbox_lists = []
        for bboxes, class_no, score in zip(bbox_lists, class_lists, scores):
            person_counter += 1
            bboxes[0] = int(bboxes[0])
            bboxes[1] = int(bboxes[1])
            bboxes[2] = int(bboxes[2])
            bboxes[3] = int(bboxes[3])

            if DRAW:
                color = (255, 0, 0)
                cv2.rectangle(resized_frame, (bboxes[0], bboxes[3]-20), (bboxes[2], bboxes[3]), (100, 100, 0), -1)
                cv2.putText(resized_frame, str(round(score, 3)), (bboxes[0] + 5, bboxes[3] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 100), 1)
                cv2.rectangle(resized_frame, (bboxes[0], bboxes[1]), (bboxes[2], bboxes[3]), color, 2)

            imp_bbox_lists.append(bboxes)
            #person_image = resized_frame[bboxes[1]: bboxes[3], bboxes[0]: bboxes[2]]
            #score = round(score, 2)
            # cv2.imwrite('images/Person_small_{}_{}.jpg'.format(person_counter, score), person_image)
            #person_image_2 = big_frame[int(2.25*bboxes[1]): int(2.25*bboxes[3]), int(2.25*bboxes[0]): int(2.25*bboxes[2])]
            #cv2.imwrite('images/Person_{}_{}.jpg'.format(person_counter, score), person_image_2)
        
        objects, object_start_time = centroid_tracker.update(imp_bbox_lists)

        # print(objects, object_start_time)

        for (objectID, centroid) in objects.items():
            text = "ID {}".format(objectID)
            obj_time = int(time.time()-object_start_time[objectID])
            if DRAW:
                pass
                # print('centroid tracker')
                cv2.rectangle(resized_frame, (centroid[0]-20, centroid[1]-30), (centroid[0]+60, centroid[1]+10), (100, 50, 100), -1)
                cv2.putText(resized_frame, text + ' T' + str(obj_time), (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
                cv2.circle(resized_frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            # bag_count = objectID

        od_tracker_count = centroid_tracker.count
        pos_count = centroid_tracker.positive_direction_count
        neg_count = centroid_tracker.negative_direction_count
        ignore_count = centroid_tracker.ignore_count

        ext_pos_count = centroid_tracker.extended_positive_direction_count
        ext_neg_count = centroid_tracker.extended_negative_direction_count
        ext_ignore_count = centroid_tracker.extended_ignore_count

        if DRAW:
            cv2.line(resized_frame, LINE[0], LINE[1], (0, 0, 255), 2)
            cv2.line(resized_frame, (ENTRY_POINT[0]-10, ENTRY_POINT[1]), (ENTRY_POINT[0]+10, ENTRY_POINT[1]), (255, 0, 0), 2)
            cv2.line(resized_frame, (ENTRY_POINT[0], ENTRY_POINT[1]-10), (ENTRY_POINT[0], ENTRY_POINT[1]+10), (255, 0, 0), 2)


            bottom_insights_img = np.zeros((100, 300, 3), dtype=np.uint8)
            bottom_insights_img[:, :] = purple_color


            # cv2.putText(bottom_insights_img, f"C   {od_tracker_count}", (150, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(bottom_insights_img, f"P   {pos_count}       {ext_pos_count}", (100, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(bottom_insights_img, f"N   {neg_count}       {ext_neg_count}", (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            # cv2.putText(bottom_insights_img, f"I   {ignore_count} {ext_ignore_count}", (150, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            for c in range(0, 3):
                bottom_insights_img[y1:y2, x1:x2, c] = (alpha_s * LOGO[:, :, c] + alpha_l * bottom_insights_img[y1:y2, x1:x2, c])
            

        
        # cv2.namedWindow('Frame', cv2.WINDOW_KEEPRATIO)
        # cv2.imshow("Frame", resized_frame)
        # cv2.namedWindow('frame-count', cv2.WINDOW_KEEPRATIO)
        if DRAW:
            concatenated_vertical_ = np.concatenate((top_insights_img, resized_frame), axis=0)
            concatenated_vertical = np.concatenate((concatenated_vertical_, bottom_insights_img), axis=0)
            cv2.imshow("frame-count", concatenated_vertical)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if VIDEO_WRITE:
            # write raw video frame to input stream of ffmpeg sub-process
            process.stdin.write(resized_frame.tobytes())


if VIDEO_WRITE:
    process.stdin.close()
    process.wait()
    process.terminate()
