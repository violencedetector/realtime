
#Teste com 64 imagens resultando na classificacao com o modelo testado no real violence scenes


import argparse
import logging
import time
import glob
import ast
import os



#import common
from tf_pose import common
import cv2
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import numpy as np

from tensorflow.keras.models import load_model

########################################
def polar(x0,y0,x1,y1):
    if x1==0 or x0==0 or y1==0 or y0==0: 
        result=0,0
    else:
        xd=x1-x0
        yd=y1-y0

        r = np.sqrt(xd**2+yd**2)
        t = np.arctan2(yd,xd)

        if t<0:
            t+=2*np.pi

        result=r,t
    return result



#centroid of point array
def centroid(points):
    list_centerx=[]
    list_centery=[]

    for p in range(points.shape[0]):
        pointx,pointy=points[p]        
        if pointx!=0 and pointy!=0:
            list_centerx.append(pointx)
            list_centery.append(pointy)

    return sum(np.array(list_centerx))/len(list_centerx),sum(np.array(list_centery))/len(list_centery)

#2 points distance
def distance(x0,y0,x1,y1):
    point1=np.array((x0,y0))
    point2=np.array((x1,y1))
    dist = np.linalg.norm(point1 - point2)
    return dist

#distances matrix based on centroid array
def distance_matrix(list_centroid):
    list_dist=[]
    list_dist_=[]
    for i,pair1 in enumerate(list_centroid):
        for n,pair2 in enumerate(list_centroid):
            x0,y0=pair1
            x1,y1=pair2 
            list_dist.append(distance(x0,y0,x1,y1))
        list_dist_.append(list_dist)
        list_dist=[]
    return list_dist_

#return second minumum in a list
def return_minumum_index(list_distances):
    #print(list_distances)
    return sorted([*enumerate(list_distances)], key=lambda x: x[1])[1][0]

########################################

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run by folder')
    parser.add_argument('--folder', type=str, default='./images/')
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--scales', type=str, default='[None]', help='for multiple scales, eg. [1.0, (1.1, 0.05)]')
    parser.add_argument('--file', type=str, default='*.jpg')
    args = parser.parse_args()
    scales = ast.literal_eval(args.scales)

    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    files_grabbed = glob.glob(os.path.join(args.folder,args.file ))#'cam149.mp4*.jpg'
    all_humans = dict()
#'cam11.mp4*.jpg_' V

    frame_counter=1
    DEPTH=64
    individuals_frame0=[]
    features_radius=[]
    features_angle=[]
    featurescx=[]
    featurescy=[]

    consolidated_features_lstm = np.zeros((2,63,72))


    for i, file in enumerate(sorted(files_grabbed)):
        # estimate human poses from a single image !
        print(file)
        image = common.read_imgfile(file, None, None)
        t = time.time()
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)
        elapsed = time.time() - t


        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        individuals=TfPoseEstimator.return_points(image, humans, imgcopy=False)

        individuals=np.array(individuals)



        #Check 2 individuals for keeping it simple
        if individuals.shape[0]==2:
            if frame_counter==1:#DEPTH:
                individuals_frame0=individuals

            else:
                for person in range(individuals.shape[0]):
                    #Clean temp arrays
                    features_radius=[]
                    features_angle=[]
                    featuresc=[]
                    featurescx=[]
                    featurescy=[]
                    consolidated = []##

                    # Calculate Azimuthal displacement between previous frame and current
                    for point in range(individuals.shape[1]):
                        x1,y1 = individuals[person][point]
                        x0,y0 = individuals_frame0[person][point] 
                        radius,angle=polar(x0,y0,x1,y1)
                        features_radius.append(radius)
                        features_angle.append(angle)

                    # Calculate centroid for this frame, first list of the distances,
                    # then distance between the point to the centroid of the nearst person
                    list_centroid = []
                    for person_centroid in range(individuals.shape[0]):
                        list_centroid.append(centroid(individuals[person_centroid]))
                        # debug
                        cx, cy = centroid(individuals[person_centroid])
                        cv2.putText(image, "X",
                                    (int(cx), int(cy)), cv2.FONT_HERSHEY_COMPLEX, 1.5,
                                    (0, 255, 0), 2)
                    list_distances = distance_matrix(list_centroid)
                    # print(list_distances)
                    featuresc=[]
                    for point_centroid in range(individuals.shape[1]):
                        x, y = individuals[person][point_centroid]
                        if x != 0 and y != 0:
                            featuresc.append(np.subtract(individuals[person][point_centroid],
                                                         list_centroid[
                                                             return_minumum_index(list_distances[person])]))
                        else:
                            featuresc.append((0, 0))
                    # print(point_centroid)
                    for featurec in featuresc:
                        tempx, tempy = featurec

                        featurescx.append(abs(tempx) / (image.shape[1] / 2))
                        featurescy.append(abs(tempy) / (image.shape[0] / 2))
                    ###########################


                    #consolidate feature for lstm
                    temp_radius=np.array(features_radius)
                    temp_angle=np.array(features_angle)
                    temp_cx=np.array(featurescx)
                    temp_cy=np.array(featurescy)


                    consolidated_features_lstm[person][frame_counter-2]=np.hstack((temp_radius,temp_angle,temp_cx,temp_cy))


            individuals_frame0=individuals

        frame_counter+=1

        if cv2.waitKey(1) == 27:
            break
        logger.debug('finished+')

    print("Model LSTM: ")
    model = load_model("best_acc_final.keras")
    predictions = model.predict(consolidated_features_lstm)
    print(np.argmax(predictions, axis=1))

    cv2.imshow('tf-pose-estimation result', image)
    while cv2.waitKey(1) != 27:            
        a=0



