import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file. Change XVID to mp4v if want to save video as .mp4')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')
flags.DEFINE_boolean('zoom', False, 'for detection improvement using zoom-in method')

# start = time.process_time()

def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    tracker = Tracker()

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width > 640 : 
        width, height = 1280, 720

    # get video ready to save locally if flag is set
    if FLAGS.output:
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, 23, (width, height))

    frame_num = 0
    frame_skip = 60
    fps_all = []
    obj = 0
    # while video is running
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.resize(frame, (width, height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num +=1

        # whole frame
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()
        if frame_num % frame_skip == 0:
            # middle left frame
            image_data1 = cv2.resize(frame, (input_size, input_size))
            image_data1 = image_data1[104:312, :208]
            image_data1 = cv2.resize(image_data1, (input_size, input_size))
            image_data1 = image_data1 / 255.
            image_data1 = image_data1[np.newaxis, ...].astype(np.float32)

            # middle right frame
            image_data2 = cv2.resize(frame, (input_size, input_size))
            image_data2 = image_data2[104:312, 208:]
            image_data2 = cv2.resize(image_data2, (input_size, input_size))
            image_data2 = image_data2 / 255.
            image_data2 = image_data2[np.newaxis, ...].astype(np.float32)

            # middle centre frame
            image_data3 = cv2.resize(frame, (input_size, input_size))
            image_data3 = image_data3[104:312, 104:312]
            image_data3 = cv2.resize(image_data3, (input_size, input_size))
            image_data3 = image_data3 / 255.
            image_data3 = image_data3[np.newaxis, ...].astype(np.float32)

        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]


            if frame_num % frame_skip == 0:
                batch_data1 = tf.constant(image_data1)
                pred_bbox1 = infer(batch_data1)
                for key, value in pred_bbox1.items():
                    lower = tf.math.greater_equal(value[:, :, 0:4],0.02)
                    upper = tf.math.less_equal(value[:, :, 0:4],0.98)
                    between = tf.reduce_all(lower&upper, axis=2)
                    small = tf.math.less_equal(value[:,:,3]-value[:,:,1], 0.25)
                    after = tf.boolean_mask(value, between & small)
                    value = tf.reshape(after, [1,-1,84])  
                    boxes1 = value[:, :, 0:4]
                    pred_conf1 = value[:, :, 4:]

                batch_data2 = tf.constant(image_data2)
                pred_bbox2 = infer(batch_data2)
                for key, value in pred_bbox2.items():
                    lower = tf.math.greater_equal(value[:, :, 0:4],0.02)
                    upper = tf.math.less_equal(value[:, :, 0:4],0.98)
                    between = tf.reduce_all(lower&upper, axis=2)
                    small = tf.math.less_equal(value[:,:,3]-value[:,:,1], 0.25)
                    after = tf.boolean_mask(value, between & small)
                    value = tf.reshape(after, [1,-1,84])  
                    boxes2 = value[:, :, 0:4]
                    pred_conf2 = value[:, :, 4:]

                batch_data3 = tf.constant(image_data3)
                pred_bbox3 = infer(batch_data3)
                for key, value in pred_bbox3.items():
                    lower = tf.math.greater_equal(value[:, :, 0:4],0.02)
                    upper = tf.math.less_equal(value[:, :, 0:4],0.98)
                    between = tf.reduce_all(lower&upper, axis=2)
                    small = tf.math.less_equal(value[:,:,3]-value[:,:,1], 0.25)
                    after = tf.boolean_mask(value, between & small)
                    value = tf.reshape(after, [1,-1,84])  
                    boxes3 = value[:, :, 0:4]
                    pred_conf3 = value[:, :, 4:]

                boxes1 = tf.add( boxes1/2, tf.constant([0.25, 0, 0.25, 0]))
                boxes2 = tf.add( boxes2/2, tf.constant([0.25, 0.5, 0.25, 0.5]))
                boxes3 = tf.add( boxes3/2, tf.constant([0.25, 0.25, 0.25, 0.25]))

                # put all frames' tensor into one
                boxes = tf.concat([boxes, boxes1, boxes2, boxes3], axis=1)
                pred_conf = tf.concat([pred_conf, pred_conf1, pred_conf2, pred_conf3], axis=1)

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        # original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, height, width)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['cow']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)

        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        detections = [Detection(bbox, score, class_name) for bbox, score, class_name in zip(bboxes, scores, names)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        # Call the tracker
        tracker.predict()
        tracker.update(detections, width, frame_num, frame_skip)

        # update tracks
        for track in tracker.tracks:
            # if not track.is_confirmed() or track.time_since_update > 1:
            #     continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            
        # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            if track.is_confirmed():
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2) 
            elif track.is_occluded(): 
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,255,255), 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame,  str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

        # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        fps_all.append(fps)
        fps_str =  "{:.2f}".format(fps)
        shapes = frame.copy()

        if FLAGS.count:
            text  =  "CowTracker; "+fps_str+" FPS"
            cv2.rectangle(shapes, (5, 5), (5 + (len(str(text)))*16, 40), (20,210,4), -1)
            alpha = 0.5
            cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0,frame)
            cv2.putText(frame,  text,(15, 30),0, 0.75, (255,255,255),2)
 
        # if frame_num % frame_skip == 0:
        obj = obj + num_objects
        print(frame_num,"object detected" , num_objects, "total", obj)
        # print(frame_num,"FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)
        
        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    fps_ave= sum(fps_all)/len(fps_all)
    print("frame num", len(fps_all), "fps ave", fps_ave, "time", frame_num/fps_ave)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
