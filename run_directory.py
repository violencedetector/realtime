
import argparse
import logging
import time
import glob
import ast
import os
#import dill


#import common
from tf_pose import common

import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

#from lifting.prob_model import Prob3dPose
#from lifting.draw import plot_pose

#import pickle#

#def write(data, outfile):
#        f = open(outfile, "w+b")
#        pickle.dump(data, f)
#        f.close()

#def read(filename):
#        f = open(filename)
#        data = pickle.load(f, encoding='latin1')
#        u = pickle._Unpickler(f)
#        u.encoding = 'latin1'
#        data = u.load()
#        f.close()
#        return data
import numpy as np

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
    args = parser.parse_args()
    scales = ast.literal_eval(args.scales)

    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    files_grabbed = glob.glob(os.path.join(args.folder, '*.jpg'))
    all_humans = dict()
    lista=[]
    counter=1
    DEPTH=64
    individuals_full=[]

    list_individuals=[]##
    individuals_reviwed=[]##
    
    total_indv=0

    #individuals_full=np.array([])

    def find_back_forward(query,index):
        result=99
        broken=False
        for internal in reversed(range(index+1)):#
            #print(internal)
            if list_individuals[internal]==query:
                result=individuals_full[internal]
                broken=True
                break
        if broken!=True:
            for internal in range(index+1,len(list_individuals),1):
                #print(internal)
                if list_individuals[internal]==query:
                    result=individuals_full[internal]
                    break
        #print(result)
        return result

#LOG FILE
    f_log = open("log_dif_dim_vio_test3b.txt", "a")#
    f_log.write("[Start]\n")#

    for i, file in enumerate(sorted(files_grabbed)):
        # estimate human poses from a single image !
        image = common.read_imgfile(file, None, None)
        t = time.time()
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)
        elapsed = time.time() - t

        logger.info('inference image #%d: %s in %.4f seconds.' % (i, file, elapsed))

        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        #cv2.imshow('tf-pose-estimation result', image)
        #cv2.waitKey(5)
        ###cv2.imwrite(file+"_",image)#SALVA IMAGENS COM O ESQUELETO
        #print(file)
##        print(humans)
        ###
        individuals=TfPoseEstimator.return_points(image, humans, imgcopy=False)

##        for individual in individuals:
##                 print("pessoa")
##                 for idx,points in enumerate(individual):
                 #for idx in range(18):
##                         print(idx,points)#individual[idx])#points)

#        write(individuals, "poses.data")
#        read_data = read("poses.data")

        individuals_full.append(individuals)
# TRATAR NUMERO DIFERENTE DE INDIVIDUOS, COMO FAZER?
# IGNORAR POUCOS PONTOS? IDENTIFICAR Q NãO é PROXIMO DOS DEMAIS?



        ##print(len(individuals))
	
        if (i % DEPTH != 0) and last_dimension!=len(individuals): 
            print(" [incompatible dimension] "+file)
            f_log.write(" [incompatible dimension] "+file+"- "+str(last_dimension)+"/"+str(len(individuals))+"\n")#
        last_dimension=len(individuals) 
        
        list_individuals.append(len(individuals))##
                

        #individuals_full=np.vstack((individuals_full,individuals))

       #np.append(individuals_full,individuals,0)
        if counter!= DEPTH:
            counter+=1
        else:
            
            most_common_item = max(list_individuals, key = list_individuals.count)##
            
            for counter_duplicate in range(DEPTH):		
#                if counter_duplicate!=0:
#                    if list_individuals[counter_duplicate]!=list_individuals[counter_duplicate-1]:#!!!!!!!comparar com o maximo comun, trocar #desde o primeiro / testar test.py no primeiro item!!!!!! o primeiro item não está comparando com o maximo
#                        print("troquei: "+str(list_individuals[counter_duplicate])+"/"+str(list_individuals[counter_duplicate-1]))
#                        individuals_reviwed.append(find_back_forward(most_common_item,counter_duplicate))
#                        print("frame: "+str(counter_duplicate))
#                        #print(len(individuals_reviwed))
#                    else:
#                        individuals_reviwed.append(individuals_full[counter_duplicate])
#                else:
#                    individuals_reviwed.append(individuals_full[counter_duplicate])

                if list_individuals[counter_duplicate]!=most_common_item:
                    ##print("troquei: "+str(list_individuals[counter_duplicate])+"item: "+str(counter_duplicate))
                    individuals_reviwed.append(find_back_forward(most_common_item,counter_duplicate))
                else:
                    individuals_reviwed.append(individuals_full[counter_duplicate])

            #print("review\n")
            #print(individuals_reviwed)

            data=np.array(individuals_full)
            with open(file+'.npy', 'wb') as f:
                ##np.save(f,data)#troquei pelo reviewed
                np.save(f,np.array(individuals_reviwed))
#            with open(file+'.npy', 'rb') as f:
#                read_data = np.load(f, allow_pickle=True)
            #print(read_data)
            #print(read_data.shape)
##############
#            f_log.write(" [normal] \n")#
#            f_log.write(str(np.array(individuals_full)))#
#            f_log.write(" [revisao] \n")#
#            f_log.write(str(np.array(individuals_reviwed)))#
#!!!!!!!!!!!!!!!!!!!!!!!!!! EXTRAIR SOMENTE OF FRAMES COM MESMA QTD DE INDIVIDUOS
#                           POSSO AUMENTAR O FRAME RATE PRA EXTRAIR MAIS FRAMES
#                            VOU TENTAR PASSAR INFRMAçAO DIRETO PRA LSTM
#                           POSSIVELMENTE UTILIZAR DISTANCIA EUCLIDIANA DE CENTROIDES
#                            OUTRA POSSIBILIDADE é IGNORAR DEPERMINADO NUMERO DE PONTOS, MAS 
#                            O MSID PROVAVEL É Q A QRD MAIS COMUN NãO TENHA MUITA SUGEIRA
            f_log.write("Qtd: "+str(most_common_item))
            f_log.write("ocorrencia: "+str(list_individuals.count(most_common_item)))

            
            total_indv+=int(list_individuals.count(most_common_item))
##############
            counter=1
            individuals_full=[]
            f_log.write(" [New file] \n")#

            list_individuals=[]##
            individuals_reviwed=[]##

#	data=np.array(individuals_full)
#	with open('test.npy', 'wb') as f:
#	         np.save(f,data)
#	with open('test.npy', 'rb') as f:
#	         read_data = np.load(f)
#	print(read_data)
        ###

#lista arqivos que pose nao foi identificada
        if humans!=[]:
                 a=1
        else:
                 lista.append(file)
        all_humans[file.replace(args.folder, '')] = humans

    f_log.write(" [total individuals found] "+str(total_indv))


#    with open(os.path.join(args.folder, 'pose.dil'), 'wb') as f:
#        dill.dump(all_humans, f, protocol=dill.HIGHEST_PROTOCOL)
    ##print(lista)
    f_log.close()#
    f.close()