from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import cv2
import os
from datetime import datetime
import time
from tarsyer import video_stream_queue as VSQ
from config import *
put_logo = False
video_write = False
show_image = True
from nms import *
from config import *
from motion_detect import motion_detection
from tarsyer.SSD_TfLite_Detector import *
from nms import non_max_suppression_fast
from centroid_check import is_point_within_polygon

camera_vsq = VSQ.VideoStreamQueue(VIDEO_INPUT, CAMERA_DETAIL, CROPPING_COORD_PRI, SKIP_FRAME, CAMERA_NO)

camera_vsq.vsq_logger.info('VSQ in progress')
camera_vsq.start()
total_frame_count = camera_vsq.stream.get(cv2.CAP_PROP_FRAME_COUNT)
print('total_frame_count = {}'.format(total_frame_count))
camera_vsq.vsq_logger.info('CODE STARTED')

motion_confirm = False
motion_thresh_counter = 0
NO_OBJECT_THRESH = 3
prev_frame_time = time.time() - 5000
MOTION_CONFIRM_THRESH = 3

# defining obj detector function for person
# object_detector = SSD_TfLite_Detection(CONFIDENCE_THRESH, MODEL_PATH)

if put_logo:
    # Define logo coordinates for the first logo
    LOGO_y1, LOGO_y2, LOGO_x1, LOGO_x2 = 100, 400, 100, 400

    # Read and resize the first logo
    LOGO1 = cv2.imread('logo.png', -1)
    LOGO1 = cv2.resize(LOGO1, (LOGO_x2 - LOGO_x1, LOGO_y2 - LOGO_y1))
    alpha_s1 = LOGO1[:, :, 3] / 255.0
    alpha_l1 = 1.0 - alpha_s1

model = YOLO("models/yolov8m.pt")
names = model.names
print(names)
# cap = cv2.VideoCapture("rtsp://admin:Admin%40123@192.168.10.66")
# assert cap.isOpened(), "Error reading video file"
# w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

save_image_dir = "tmp/alert_images"
save_image_dir_for_email = "tmp/alert_images_for_email"
if not os.path.exists(save_image_dir):
    os.mkdir(save_image_dir)
if not os.path.exists(save_image_dir_for_email):
    os.mkdir(save_image_dir_for_email)

# if video_write:
#     # Video writer
#     video_writer = cv2.VideoWriter("processed_awarpur_safety_with-white.avi",
#                                    cv2.VideoWriter_fourcc(*'mp4v'),
#                                    fps, (w, h))
def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """

    if label == 0:
        color = (145, 145, 255)
    else:
        color = [200, 200, 200]
    return tuple(color)


skip_frames = 20
skip_counter = 0
frame_counter = 0
idx = 0
no_detection_output_counter = 0
Intrusion_entry_points = np.array([[23, 61], [208, 38], [295, 136], [284, 233], [94, 287], [60, 270], [23, 61]], np.int32)
send_alert_time = time.monotonic() - 60
send_alert_interval = 30
while True:
    curr_time = time.time()
    camera_dict = camera_vsq.read()
    camera_status = camera_dict['camera_status']
    # ret, frame = video_cap.read()
    processing_start_time = time.time()
    if camera_status:
        resized_frame, big_frame = camera_dict['image']
        # im0 = im_ori[136:1520, 544:1928]
        im0 = resized_frame

        if curr_time - prev_frame_time > 60:
            prev_frame_time = curr_time
            base_frame = resized_frame.copy()
            print('saving previous frame.')
            camera_vsq.vsq_logger.info('Changing previous frame for motion detector')
        frame_counter += 1
        # Draw the Roi on resized frame
        cv2.polylines(resized_frame, [Intrusion_entry_points], True, (255), 2)

        if not motion_confirm:
            cv2.putText(resized_frame, "Motion detector", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 100), 1)
            returned_status, _, _ = motion_detection(base_frame, resized_frame, MIN_CONTOUR_AREA)

            if returned_status:
                motion_thresh_counter += 1
                print("Motion Detected")

                if motion_thresh_counter == MOTION_CONFIRM_THRESH:
                    motion_thresh_counter = 0
                    motion_confirm = True
                    camera_vsq.vsq_logger.info('Universal: Motion Detected.')

            else:
                motion_thresh_counter = 0

        else:
            m_time = time.monotonic()
            motion_detect_start_time = time.monotonic()
            inference_start_time = time.time()
            results = model.predict(im0, show=False, iou=0.1, conf=0.1, classes=0)
            inference_end_time = time.time()
            inference_time = inference_end_time - inference_start_time
            print(f"Inference Time: {inference_time} seconds")
            boxes = results[0].boxes.xyxy.cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()
            print(clss)

            if len(boxes) == 0:
                no_detection_output_counter += 1
                if no_detection_output_counter >= NO_OBJECT_THRESH:
                    no_detection_output_counter = 0
                    motion_confirm = False
                    base_frame = resized_frame.copy()
                    camera_vsq.vsq_logger.info('Universal: Motion confirm False')

            elif len(boxes) > 0:
                print("Person Detector On")

                motion_detect_start_time = time.monotonic()
                CENTROID_CHECK = False
                if boxes is not None:
                    for box, cls in zip(boxes, clss):
                        idx += 1
                        x1, y1, x2, y2 = box
                        if x2-x2 < 200 and y2-y1 < 200:
                            color = compute_color_for_labels(cls)
                            # cv2.rectangle(im0, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2, )
                            cls = int(cls)

                            cv2.rectangle(im0, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2, )

                            centroid_x = (x1 + x2) / 2
                            centroid_y = (y1 + y2) / 2
                            centroid = (centroid_x, centroid_y)
                            CENTROID_CHECK = is_point_within_polygon(Intrusion_entry_points, centroid)
                            print("Centroid check -----", CENTROID_CHECK)
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                    if CENTROID_CHECK:
                        if abs(send_alert_time - time.monotonic()) > send_alert_interval:
                            send_alert_time = time.monotonic()
                            image_name = f"{timestamp}_Intrusion.jpg"

                            # filename = f"{save_image_dir}/{image_name}"
                            filename_for_email = f"{save_image_dir_for_email}/{image_name}"

                            # cv2.imwrite(filename, big_frame)
                            cv2.imwrite(filename_for_email, big_frame)

        if put_logo:
            # Blend the first logo onto the frame
            for c in range(0, 3):
                im0[LOGO_y1:LOGO_y2, LOGO_x1:LOGO_x2, c] = (
                        alpha_s1 * LOGO1[:, :, c] + alpha_l1 * im0[LOGO_y1:LOGO_y2, LOGO_x1:LOGO_x2, c]
                )
        if show_image:
            cv2.namedWindow('resized_frame', cv2.WINDOW_KEEPRATIO)
            cv2.imshow("resized_frame", im0)


        processing_end_time = time.time()
        processing_time = processing_end_time - processing_start_time
        print(f"processing Time: {processing_time} seconds")
        # if video_write:
        #     video_writer.write(im0)

        if show_image:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# cap.release()
# if video_write:
#     video_writer.release()
if show_image:
    cv2.destroyAllWindows()

