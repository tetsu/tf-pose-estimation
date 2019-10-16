import argparse
import csv
import logging
import time
import os

import cv2
import numpy as np
import pafy
import pandas as pd

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

start_time = time.time()

logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--saveas', type=str, default=False)
    parser.add_argument('--youtube', type=bool, default=False,
                        help='True to stream from Youtube.')
    parser.add_argument('--resolution', type=str, default='432x368',
                        help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin',
                        help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--showBG', type=bool, default=True,
                        help='False to show skeleton only.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' %
                 (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    if args.youtube:
        url = args.video
        vPafy = pafy.new(url)
        # play = vPafy.getbest(preftype="webm")
        play = vPafy.getbest().url
        # cap = cv2.VideoCapture(play.url)
        cap = cv2.VideoCapture(play)
    else:
        cap = cv2.VideoCapture(args.video)
    
    print('FPS: ' + str(cap.get(cv2.CAP_PROP_FPS)))

    if cap.isOpened() is False:
        print("Error opening video stream or file")

    # Create Pandas DataFrame, save as pose.csv
    columns = ['fps', 'time']
    for i in range(18):
        columns.append(str(i)+'x')
        columns.append(str(i)+'y')
    
    df = pd.DataFrame(columns=columns)
    df.index.name = 'index'
    frame_interval = 1 / cap.get(cv2.CAP_PROP_FPS)
    
    frame_index = 0
    while cap.isOpened():
        try:
            ret_val, image = cap.read()
            # set resolution
            # image = cv2.resize(image, dsize=(432, 368))
            image = cv2.resize(image, dsize=(480, 270))
            # if args.resolution:
            #     size = args.resolution.split('x')
            #     image = cv2.resize(image, dsize=(int(size[0]), int(size[1])))
            humans = e.inference(image, resize_to_default=True, upsample_size=4.0)

            # save keypoints
            if len(humans) > 0:
                human = humans[0]

                data_series = {
                    'fps': str(cap.get(cv2.CAP_PROP_FPS)),
                    'time': str(frame_interval * frame_index)
                }
                for k in human.body_parts:
                    body_part = human.body_parts[k]
                    data_series[str(body_part.part_idx)+'x'] = body_part.x
                    data_series[str(body_part.part_idx)+'y'] = body_part.y
                
                image_keypoints = pd.Series( data_series, index=df.columns, name=str(frame_index))
                df = df.append(image_keypoints)
            
            if not args.showBG:
                image = np.zeros(image.shape)
            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

            cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)),
                        (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow('tf-pose-estimation result', image)
            fps_time = time.time()
            if cv2.waitKey(1) == 27:
                break
        except:
            pass
            break

        frame_index = frame_index + 1
    
    cv2.destroyAllWindows()

    # save result as CSV file
    if args.saveas:
        if not os.path.exists(os.path.dirname(args.saveas)):
            os.mkdir(os.path.dirname(args.saveas))
        df.to_csv(args.saveas)
    else:
        if not os.path.exists('./videos/csv'):
            os.mkdir('./videos/csv')
        filename, _ = os.path.splitext(os.path.basename(args.video))
        df.to_csv('./videos/csv/' + filename + '-' + args.model + '.csv')

logger.debug('finished in ' + str(time.time() - start_time) + ' seconds')