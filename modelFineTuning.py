import os.path

import tensorflow as tf
import numpy as np
from modelPreTraining import *
from gestureDataLoader import signDataLoader,WidarDataloader
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Softmax, Dense, Reshape, Lambda,Dot,concatenate,ZeroPadding2D,Conv2D,\
    MaxPooling2D,Flatten,Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.metrics.pairwise import cosine_similarity
from Config import getConfig
# from MODEL import models
import matplotlib.pyplot as plt
import random
from scipy.io import savemat,loadmat
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import time
from methodTesting.t_SNE import *
# from modelPreTraining import PreTrainModel

seed = 42
import random
import numpy as np

random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


class fineTuningSignFi:
    def __init__( self,config, isZscore = False, isiheritance = False ):
        self.nshots = config.nshots
        self.isZscore = isZscore
        self.modelObj = models( )
        self.config = config
        self.trainTestObj = PreTrainModel(config = config )
        self.lrScheduler = tf.keras.callbacks.LearningRateScheduler( self.trainTestObj.scheduler )
        # self.lrScheduler = ReduceLROnPlateau(
        #         monitor = 'val_loss', factor = 0.1,
        #         patience = 20,
        #         )
        self.earlyStop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 15, restore_best_weights
        =True, min_delta = 0.0001/2,mode = 'min',verbose=1)
        # self.pretrained_featureExtractor = self._getPreTrainedFeatureExtractor( )
        # self.pretrained_featureExtractor.trainable = True
        if self.nshots == 1:
            self.isOneShotTask = True
        else:
            self.isOneShotTask = False
    def _getSQData( self,nshots:int ):
        '''
        This function build for split support set and query set according to the number of shots
        :param nshots:
        :return:
        '''
        testSign = signDataLoader( config = self.config )

        if type(self.config.source) == list or 'user' in self.config.source:
            _, _, x_test, y_test = testSign.getFormatedData( source = self.config.source,isZscore = self.isZscore )
            x = x_test[ 1250:1500 ]
            y = y_test[ 1250:1500 ]
            num = nshots * 25
            Support_data = x[ 0:num, :, :, : ]
            Support_label = y[0:num,:]
            Query_data = x[num:len(x)+1,:,:,:]
            Query_label = y[num:len(x)+1,:]

            # n = 5
            # Support_data = x[ 25*n:25*(n+1), :, :, : ]
            # Support_label = y[25*n:25*(n+1),:]
            # Query_data = np.delete(x,np.arange(25*n,25*(n+1)),0)
            # Query_label = np.delete(y,np.arange(25*n,25*(n+1)),0)
        elif 'home' in self.config.source or 'lab' in self.config.source:
            _, _, x_test, y_test, = testSign.getFormatedData( source = self.config.source,isZscore=self.isZscore )
            num = nshots * (np.max(y_test) + 1 - np.min(y_test))
            Support_data = x_test[ 0:num, :, :, : ]
            Support_label = y_test[ 0:num, : ]
            Query_data = x_test[ num:len( x_test ) + 1, :, :, : ]
            Query_label = y_test[ num:len( x_test ) + 1, : ]
        output = {'Support_data':Support_data,
                  'Support_label':Support_label,
                  'Query_data':Query_data,
                  'Query_label':Query_label}
        return output
    def _getValData( self,Query_data,Query_label ):
        '''
        Get the validation data for fine tuning
        :return:
        '''
        # val_data = self.data['Query_data']
        # val_label = to_categorical(
        #         self.data[ 'Query_label' ] - np.min( self.data[ 'Query_label' ] ), num_classes =
        #         self.config.N_novel_classes
        #         )
        val_data = Query_data
        val_label = to_categorical(
                Query_label - np.min( Query_label ), num_classes =
                self.config.n_ft_cls
                )
        return [val_data,val_label]
    def _getDataToTesting(self,query_set,nway,mode:str = 'fix'):
        '''
        Randomly choose one sample from the Query data set
        :param query_set:
        :param nway:
        :return:
        '''
        if mode == 'fix':
            sample_sign = np.random.choice(np.arange(0,len(query_set),self.config.N_novel_classes ),size = 1,
                    replace = False)
            sample_index = random.randint( 0, nway - 1 )
            query_data = np.repeat( query_set[ sample_sign+sample_index ], [ nway ], axis = 0 )
            return [ query_data, sample_index ]
        elif mode == 'random':
            sample_sign = np.random.choice(
                    np.arange( 0, len( query_set ), self.config.N_novel_classes ), size = 2, replace = False
                    )
            sample_index = random.randint( 0, nway - 1 )
            support_data = np.repeat( query_set[ sample_sign[0] + sample_index ], [ nway ], axis = 0 )
            query_data = np.repeat( query_set[ sample_sign[1] + sample_index ], [ nway ], axis = 0 )
            return [ support_data,query_data, sample_index ]
    def _getNShotsEmbedding( self,featureExtractor, Support_data):
        Sign_class = np.arange( 0, self.config.N_novel_classes, 1 )
        # Sign_samples = np.arange( 0, 125, 25 )
        Sign_samples = np.arange( 0, len(Support_data), self.config.N_novel_classes )
        n_shots_support_embedding = [ ]
        for i in Sign_class:
            n_shots_support_data = [ ]
            for j in Sign_samples:
                n_shots_support_data.append( Support_data[ i + j ] )
            n_shots_support_embedding.append(np.mean( featureExtractor.predict( np.asarray( n_shots_support_data ) ), axis = 0 ) )
        n_shots_support_embedding = np.asarray( n_shots_support_embedding )
        return n_shots_support_embedding
    def _getPreTrainedFeatureExtractor( self ):
        '''
        This function build for recreating the feature extractor and load pre-trained weights
        :return: feature extractor
        '''

        trained_featureExtractor = self.modelObj.buildFeatureExtractor( mode='Alexnet' )
        trained_featureExtractor.load_weights(self.config.pretrainedfeatureExtractor_path )
        return trained_featureExtractor
    def _loadFineTunedModel(self,applyFinetunedModel:bool = True, useWeightMatrix:bool = False,isSepRx:bool = False):
        '''
        This function build for load fine tuned model for testing
        :returns pre-trained feature extractor and fine tuned classifier
        '''
        './models/fine_tuning_signfi/fc_fineTuned_250Cls_labTohome_1_shot_without_Zscore.h5'
        if applyFinetunedModel:

            fine_Tune_model = self.modelObj.buildTuneModel( config = self.config,isTest = True )
            # fine_Tune_model = self.modelObj.buildFeatureExtractor(  )
            if not isSepRx:
                print( f'loading fine tuned model: {self.config.tunedModel_path}' )
                print('loading fine tuned model weights}')
                fine_Tune_model.load_weights(self.config.tunedModel_path)
            if useWeightMatrix:
                feature_extractor = fine_Tune_model
            else:
                feature_extractor = Model(
                        inputs = fine_Tune_model.input, outputs = fine_Tune_model.get_layer( 'lambda_layer' ).output
                        )
        elif not applyFinetunedModel:
            print( f'loading original pretrained feature extractor: {self.config.pretrainedfeatureExtractor_path}' )
            feature_extractor = self.modelObj.buildFeatureExtractor(mode = 'Alexnet')
            feature_extractor.load_weights(self.config.pretrainedfeatureExtractor_path)
        '''
        Classifier input: two feature vector
                      output: one probability
        '''
        # cls_intput_Support = Input(feature_extractor.output.shape[1],name = 'Support_input')
        # cls_intput_Query = Input( feature_extractor.output.shape[1], name = 'Query_input' )
        cls_intput_Support = Input(1280,name = 'Support_input')
        cls_intput_Query = Input( 1280, name = 'Query_input' )
        cosSim_layer = Dot( axes = 1, normalize = True )([cls_intput_Support,cls_intput_Query])
        cls_output = Softmax( )( tf.squeeze(cosSim_layer,-1) )
        classifier = Model(inputs = [cls_intput_Support,cls_intput_Query],outputs = cls_output)
        # feature_extractor, classifier = self._configModel(model = self.fine_Tune_model)
        return [feature_extractor, classifier]
    def n_tuning_classes(self,n,data):
        y = np.unique(data['Support_label'])
        FT_label = y[0:n]
        test_label = y[n:len(y)]
        support_classes_idx = np.where( data['Support_label'] == FT_label )[0]
        Query_classes_idx = np.where( data['Query_label'] == FT_label )[0]

        data['Support_data'] = data['Support_data'][support_classes_idx]
        data['Support_label'] = data['Support_label'][support_classes_idx]
        data['Query_data'] = data['Query_data'][Query_classes_idx]
        data['Query_label'] = data['Query_label'][Query_classes_idx]
        return data
    def tuning( self ,init_weights = True,init_bias = False):
        self.pretrained_featureExtractor = self._getPreTrainedFeatureExtractor( )

        # self.pretrained_featureExtractor = Model( inputs = self.pretrained_featureExtractor.input, outputs = self.pretrained_featureExtractor.get_layer( 'Maxpool_3' ).output )
        self.pretrained_featureExtractor.trainable = True
        self.data = self._getSQData( nshots = self.nshots )
        # self.data = self.n_tuning_classes(self.config.n_ft_cls,self.data)
        if 0:
            sup_label_id = np.unique( self.data["Support_label"] )
            # sup_selected_label_id = np.random.choice(sup_label_id,6,replace = False )
            sup_selected_label_id = sup_label_id[0:6]
            sup_idx = np.where( self.data[ 'Support_label' ] == sup_selected_label_id )[ 0 ]
            self.data['Support_data'] = self.data['Support_data'][sup_idx]
            self.data['Support_label'] = self.data['Support_label'][sup_idx]


            que_idx = np.where( self.data[ 'Query_label' ] == sup_selected_label_id )[ 0 ]
            self.data['Query_data'] = self.data['Query_data'][que_idx]
            self.data['Query_label'] = self.data['Query_label'][que_idx]

        fine_Tune_model = self.modelObj.buildTuneModel(
                pretrained_feature_extractor = self.pretrained_featureExtractor,
                isTest = False,config = self.config
                )
        if init_weights:
            if self.nshots == 1:
                weights = np.transpose( self.pretrained_featureExtractor.predict( self.data[ 'Support_data' ] ) )
            elif self.nshots > 1:
                weights = np.transpose(self._getNShotsEmbedding( self.pretrained_featureExtractor,self.data[ 'Support_data' ] ) )
            if init_bias:
                p = fine_Tune_model.predict(self.data['Query_data'])
                bias = np.tile(np.mean(-np.sum( p * np.log(p ),axis = 1 ) ),self.config.N_novel_classes )
            else:
                bias = np.zeros(self.config.N_novel_classes )
            fine_Tune_model.get_layer( 'fine_tune_layer' ).set_weights( [ weights, bias ] )

        val_data, val_label = self._getValData(self.data['Query_data'],self.data['Query_label'] )
        optimizer = tf.keras.optimizers.Adam(
                learning_rate = self.config.lr,
                epsilon = 1e-06,
                )
        # optimizer = tf.keras.optimizers.SGD(
        #         learning_rate = self.config.lr,
        #         momentum = 0.9,
        #         )
        fine_Tune_model.compile( loss='categorical_crossentropy', optimizer=optimizer, metrics='acc' )
        idx = np.random.permutation(len(self.data[ 'Support_data' ]))
        start = time.time( )
        self.config.history = fine_Tune_model.fit(
                self.data[ 'Support_data' ][ idx ], to_categorical( self.data[ 'Support_label' ][ idx ] - np.min(
                                self.data[ 'Support_label' ]),num_classes = self.config.n_ft_cls ),
                epochs = 1000,
                # shuffle = True,
                verbose = 0,
                validation_data = (val_data, val_label),
                callbacks = [ self.earlyStop, self.lrScheduler ]
                )
        end = time.time( )
        print( f'feature extractor tuning time is: {end - start:.2f}' )
        return fine_Tune_model
    def _mapToNways(self,Support_data,query_set,query_label,nway):
        query_label = np.argmax( query_label, axis = 1 )
        label = np.unique(query_label)
        selected_sign = np.unique(np.random.choice(label,size = nway,replace=False))
        Support_data = Support_data[selected_sign,:,:,:]
        # selected_data_idx = np.where(query_label == selected_sign)
        index = np.random.choice(np.arange(0,len(query_set),self.config.N_novel_classes ),size = 1,
                    replace = False)
        query_data = query_set[index+selected_sign,:,:,:]
        sample_index = random.randint( 0, nway - 1 )
        Query_data = np.repeat(np.expand_dims(query_data[sample_index],axis=0),[nway],axis = 0)
        return [ Support_data, Query_data,sample_index]
    def test( self, nway,applyFinetunedModel:bool=True ):
        # self.pretrained_featureExtractor = self._getPreTrainedFeatureExtractor( )
        # self.pretrained_featureExtractor.trainable = False
        # softmax_func = tf.keras.layers.Softmax( )
        self.data = self._getSQData( nshots = self.nshots )
        N_test_sample = 100
        feature_extractor, classifier = self._loadFineTunedModel( applyFinetunedModel )
        # load Support and Query dataset
        query_set, query_label = self._getValData(self.data['Query_data'],self.data['Query_label'] )
        Support_data = self.data[ 'Support_data' ]
        Support_label = self.data[ 'Support_label' ]
        test_acc = [ ]
        for i in range( 74,76 ):
            nway = i
            correct_count = 0
            print( f'................................Checking {nway} ways accuracy................................' )
            if self.isOneShotTask:
                for i in range(N_test_sample):
                    Selected_Support_data, Selected_Query_data, sample_index = self._mapToNways(Support_data,query_set,
                            query_label,nway)
                    Support_set_embedding = feature_extractor.predict( Selected_Support_data )
                    # Query_data, sample_index = self._getDataToTesting( query_set = query_set, nway = nway )
                    Query_set_embedding = feature_extractor.predict( Selected_Query_data )
                    prob_classifier = classifier.predict([Support_set_embedding,Query_set_embedding])
                    if np.argmax( prob_classifier ) == sample_index:
                        correct_count += 1
                        print( f'The number of correct: {correct_count}, The number of test count {i}' )
                acc = (correct_count / N_test_sample) * 100.
                test_acc.append( acc )
                print( "Accuracy %.2f" % acc )
            if not self.isOneShotTask:
                Support_set_embedding = self._getNShotsEmbedding( feature_extractor,Support_data )
                for i in range(N_test_sample):
                    Query_data, sample_index = self._getDataToTesting( query_set = query_set, nway = nway )
                    Query_set_embedding = feature_extractor.predict( Query_data )
                    prob_classifier = classifier.predict( [ Support_set_embedding, Query_set_embedding ] )
                    if np.argmax( prob_classifier ) == sample_index:
                        correct_count += 1
                        print( f'The number of correct: {correct_count}, The number of test count {i}' )
                acc = (correct_count / N_test_sample) * 100.
                test_acc.append( acc )
                print( "Accuracy %.2f" % acc )
        return test_acc
class fineTuningWidar(fineTuningSignFi ):
    def __init__( self,config,isMultiDomain:bool = False,isiheritance=False ):
        super().__init__(config = config, isiheritance = True, )
        self.isMultiDomain = isMultiDomain
        # self.config = config
        if not isiheritance:
            self.WidarDataLoaderObj = WidarDataloader(config = config, isMultiDomain = True )
        # self.selected_gesture_samples_data,self.x,self.y = self.WidarDataLoaderObj.x,self.WidarDataLoaderObj.x,self.WidarDataLoaderObj.y
            self.config = config
            self.nshots = config.nshots
            self.nshots_per_domain = config.nshots_per_domain
            # self.nshots_per_domain = int(self.nshots/5)
            self.nways = config.N_novel_classes
            self.pretrained_featureExtractor = self._getPreTrainedFeatureExtractor( )
            self.pretrained_featureExtractor.trainable = True
            self.initializer = tf.keras.initializers.RandomUniform( minval = 0., maxval = 1. )
            self.fine_Tune_model = self.modelObj.buildTuneModel(
                    pretrained_feature_extractor = self.pretrained_featureExtractor,
                    isTest = False, config = self.config
                    )
            if isMultiDomain:
                self.WidarDataLoaderObjMulti = WidarDataloader(
                        isMultiDomain = isMultiDomain,
                        config = config
                        )
            else:
                self.WidarDataLoaderObj = WidarDataloader(
                        isMultiDomain = isMultiDomain,
                        config = config
                        )

    def _getoutput( self, feature_extractor ):
        return Model(inputs = feature_extractor.input,outputs = feature_extractor.get_layer('lambda_layer').output)
    def _getNShotsEmbedding( self,feature_extractor,Support_set ):
        Support_set_embedding_all = feature_extractor.predict( Support_set )
        Support_set_embedding = []
        if self.isMultiDomain:
            n = len(self.config.domain_selection)
            for i in range( self.nways ):
                Support_set_embedding.append(
                        np.mean(
                                Support_set_embedding_all[ i * n * self.nshots:i * n * self.nshots + n * self.nshots ],
                                axis = 0
                                )
                        )
        else:
            # n=self.nshots
            for i in range( self.nways ):
                Support_set_embedding.append(
                        np.mean(
                                Support_set_embedding_all[ i * self.nshots:i * self.nshots + self.nshots ],
                                axis = 0
                                )
                        )

        return np.asarray(Support_set_embedding)
    def tuning( self,init_weights = True,init_bias = False,isTest:bool = False,num_sample_per_gesture = 20):
        self.data = self.WidarDataLoaderObj.getSQDataForTest(
                nshots = self.nshots, mode = 'fix',
                isTest = isTest,
                Best = None,num_sample_per_gesture = num_sample_per_gesture
                )
        # self.data = self.WidarDataLoaderObjMulti.getMultiDomainSQDataForTest(
        #         nshots_per_domain = self.nshots_per_domain, isTest = False
        #         )
        self.pretrained_featureExtractor.load_weights(self.config.pretrainedfeatureExtractor_path)
        if init_weights:
            if self.nshots == 1:
                weights = np.transpose( self.pretrained_featureExtractor.predict( self.data[ 'Support_data' ] ) )
            elif self.nshots > 1:
                weights = np.transpose(
                        self._getNShotsEmbedding( self.pretrained_featureExtractor, self.data[ 'Support_data' ] )
                        )
            if init_bias:
                p = self.fine_Tune_model.predict( self.data[ 'Query_data' ] )
                bias = np.tile( np.mean( -np.sum( p * np.log( p ), axis = 1 ) ), self.config.N_novel_classes )
            else:
                bias = np.zeros( self.config.N_novel_classes )
            self.fine_Tune_model.get_layer( 'fine_tune_layer' ).set_weights( [ weights, bias ] )
            # self.config.weight = self.fine_Tune_model.get_layer( 'FC_1' ).get_weights( )
        val_data, val_label = self.data[ 'Val_data' ], to_categorical(
                self.data[ 'Val_label' ], num_classes
                = self.config.N_novel_classes
                )
        optimizer = tf.keras.optimizers.Adam(
                learning_rate = self.config.lr,
                # momentum = 0.9,
                epsilon = 1e-06,
                )

        self.fine_Tune_model.compile( loss = 'categorical_crossentropy', optimizer = optimizer, metrics = 'acc' )
        idx = np.random.permutation( len( self.data[ 'Support_data' ] ) )
        history = self.fine_Tune_model.fit(
                self.data[ 'Support_data' ][ idx ], to_categorical(self.data[ 'Support_label' ][ idx ] , num_classes
                = self.config.N_novel_classes ),
                epochs = 1000,
                # shuffle = True,
                validation_data = (val_data, val_label),
                callbacks = [ self.earlyStop, self.lrScheduler ]
                )
        return [self.fine_Tune_model,self.data['record'],history]
    def tuningMultiRx( self ):
        data = self.WidarDataLoaderObj.getMultiDomainSQDataForTest( nshots_per_domain = self.config.nshots, isTest = False )
        Support_data = [ ]
        keys = list( data[ 'Support_data' ].keys( ) )
        [ Support_data.append( np.concatenate( data[ 'Support_data' ][ keys[ j ] ], axis = 0 ) ) for j in
          range( len( keys ) ) ]
        s_data_array = np.concatenate( Support_data, axis = 0 )

        self.pretrained_featureExtractor.load_weights( self.config.pretrainedfeatureExtractor_path )
        start = time.time( )
        weights = np.transpose(
                self._getNShotsEmbedding( self.pretrained_featureExtractor, s_data_array )
                )
        end = time.time( )
        timecost = (end - start)
        print( f'The inference time 1 is: {timecost:.2f}' )
        bias = np.zeros( self.config.N_novel_classes )
        self.fine_Tune_model.get_layer( 'fine_tune_layer' ).set_weights( [ weights, bias ] )
        optimizer = tf.keras.optimizers.Adam(
                learning_rate = self.config.lr,
                # momentum = 0.9,
                epsilon = 1e-06,
                )
        self.fine_Tune_model.compile( loss = 'categorical_crossentropy', optimizer = optimizer, metrics = 'acc' )
        val_data, val_label = data[ 'Val_data' ], to_categorical(
                data[ 'Val_label' ], num_classes
                = self.config.N_novel_classes
                )

        idx = np.random.permutation( len( s_data_array ) )
        start = time.time( )
        history = self.fine_Tune_model.fit(
                s_data_array[ idx ],
                to_categorical(data[ 'Support_label' ][ idx ], num_classes = self.config.N_novel_classes),
                epochs = 1000,
                verbose = 0,
                # shuffle = True,
                validation_data = (val_data, val_label),
                callbacks = [ self.earlyStop, self.lrScheduler ]
                )
        end = time.time( )
        timecost = (end - start)
        print( f'The inference time 2 is: {timecost:.2f}' )
        return [self.fine_Tune_model,data['record'],history]
    def test( self, applyFinetunedModel:bool):
        self.feature_extractor, self.classifier = self._loadFineTunedModel(
                applyFinetunedModel = applyFinetunedModel,useWeightMatrix = True
                )
        print(f'check for {self.nshots} shots '
              f'accuracy......................................................................')
        N_test_sample = 600
        correct_count = 0
        test_acc = [ ]
        y_true = []
        y_pred = []
        label_true = []
        n = 6
        feature_extractor = self._getoutput(self.feature_extractor)
        classifier = self.classifier
        # Support_set_embedding = np.transpose(self.feature_extractor.get_layer('fine_tune_layer').get_weights()[0])
        # Support_set_embedding = feature_extractor.predict()
        for i in range( N_test_sample ):
            self.data = self.WidarDataLoaderObj.getSQDataForTest(
                    nshots = self.nshots, mode = 'fix',
                    isTest = True, Best = self.config.record
                    )
            # self.data = self.WidarDataLoaderObjMulti.getMultiDomainSQDataForTest(
            #         nshots_per_domain = self.nshots_per_domain, isTest = True, Best = config.record
            #         )
            # Support_set_embedding = matrix
            Query_set = self.data['Query_data']
            Support_set = self.data['Support_data']
            if applyFinetunedModel:
                Support_set_embedding = np.transpose(
                        self.feature_extractor.get_layer( 'fine_tune_layer' ).get_weights( )[ 0 ]
                        )
            else:
                Support_set_embedding = self._getNShotsEmbedding(feature_extractor,Support_set)
            gesture_type_idx = i%6
            Query_sample = np.repeat( np.expand_dims( Query_set[ gesture_type_idx ], axis = 0 ), n, axis = 0 )
            Query_set_embedding = feature_extractor.predict( Query_sample )
            # model = self._getoutput( feature_extractor )
            prob_classifier = classifier.predict( [ Support_set_embedding, Query_set_embedding ] )
            y_true.append(gesture_type_idx)
            label_true.append(self.data['Query_label'][gesture_type_idx][0])
            y_pred.append( np.argmax(prob_classifier))
            if np.argmax( prob_classifier ) == gesture_type_idx:
                correct_count += 1
            print( f'The number of correct: {correct_count}, The number of test count {i}' )
        acc = (correct_count / N_test_sample) * 100.
        test_acc.append( acc )
        print( "Accuracy %.2f" % acc )
        return test_acc,[y_true,y_pred],label_true
    def testMultiRx( self,applyFinetunedModel:bool,useWeightMatrix:bool = True,n_Rx:int = 6):
        # n_Rx = 5
        self.feature_extractor, _ = self._loadFineTunedModel(
                applyFinetunedModel = applyFinetunedModel,useWeightMatrix = useWeightMatrix)
        print(f'check for {self.nshots} shots {n_Rx} Receivers'
              f'accuracy......................................................................')
        N_test_sample,correct_count,test_acc,y_true,y_pred,label_true, n = 600,0,[],[],[],[],6
        softmax_func = tf.keras.layers.Softmax( )
        if useWeightMatrix:
            weights = np.transpose(
                    self.feature_extractor.get_layer( 'fine_tune_layer' ).get_weights( )[ 0 ]
                    )
            feature_extractor = self._getoutput( self.feature_extractor )
        for i in range( N_test_sample ):
            Query_data_selection = []
            data = self.WidarDataLoaderObj.getMultiDomainSQDataForTest( 1, False )
            Support_data,Query_data,Support_embedding,Query_embedding,sim,sim_mean = [],[],[],[],[],[]
            keys = list( data[ 'Support_data' ].keys( ) )
            [Support_data.append( np.concatenate( data[ 'Support_data' ][ keys[ j ] ], axis = 0 )) for j in range(len(keys))]
            # random selection of specific receiver
            selected_Rx_idx = np.unique(np.random.choice(np.arange(0,6),n_Rx,replace = False))
            # selection of specific receiver
            # selected_Rx_idx = np.arange(0,n_Rx)
            for g in range(len(Support_data)):
                Support_data[g] = Support_data[g][selected_Rx_idx]
            if not useWeightMatrix:
                feature_extractor = self.feature_extractor
                '''gestures, 6 receivers CSI embedding'''
                [Support_embedding.append(feature_extractor.predict(Support_data[j])) for j in range(len(Support_data))]
            g_idx = np.random.choice( np.arange( 0, len( keys ) ), 1, replace = False )[ 0 ]
            sample_idx = np.random.choice( np.arange( 0, len(data['Query_data'][keys[g_idx]][0]) ), 1,replace = False)[ 0 ]
            buffer = data['Query_data'][keys[g_idx]]
            '''select one sample for the selected gesture, corresponding to six receivers'''
            [Query_data.append(buffer[ant][sample_idx]) for ant in range(len(data['Query_data'][keys[g_idx]]))]
            [Query_data_selection.append(Query_data[ant]) for ant in selected_Rx_idx]
            Query_embedding.append( feature_extractor.predict( np.asarray( Query_data_selection ) ) )
            if useWeightMatrix:
                p = []
                buf_sim = [ cosine_similarity( weights, np.expand_dims( Query_embedding[ 0 ][ ant ], axis = 0 ) )
                            for ant in range( len(Query_embedding[0])) ]
                [p.append(np.expand_dims(softmax_func( np.squeeze(buf_sim[tt],axis=-1) ).numpy( ),axis = 0 )) for tt in range(len(buf_sim) )]
                p = np.sum(np.concatenate( p, axis = 0 ),axis=0)
            else:
                for g_sim in range(len(Support_embedding ) ):
                    [sim.append(cosine_similarity( np.expand_dims(Support_embedding[ g_sim ][ ant ],axis=0 ),np.expand_dims(
                            Query_embedding[0][ant],axis=0) )) for ant in range(len(Query_embedding[0]))]
                    sim_mean.append(np.mean(sim))
                p = softmax_func(sim_mean).numpy( )
            if np.argmax( p ) == g_idx:
                correct_count += 1
        acc = (correct_count / N_test_sample) * 100.
        print( "Accuracy %.2f" % acc )
        return acc
    def GetparamForMulRx(self,path_idx:int):
        FE, _ = self._loadFineTunedModel(
                applyFinetunedModel = True, useWeightMatrix = True, isSepRx = True
                )
        FE.load_weights( self.config.tunedModel_path[ path_idx ] )
        feature_extractor = self._getoutput( FE )
        weights = np.transpose(
                FE.get_layer( 'fine_tune_layer' ).get_weights( )[ 0 ]
                )
        return feature_extractor,weights
    def testMultiRxSep( self, N_Rx:int ):
        f1, w1 = self.GetparamForMulRx( path_idx = 0 )
        f2, w2 = self.GetparamForMulRx( path_idx = 1 )
        f3, w3 = self.GetparamForMulRx( path_idx = 2 )
        f4, w4 = self.GetparamForMulRx( path_idx = 3 )
        f5, w5 = self.GetparamForMulRx( path_idx = 4 )
        f6, w6 = self.GetparamForMulRx( path_idx = 5 )
        feature_extractor = [f1,f2,f3,f4,f5,f6]
        weights=[w1,w2,w3,w4,w5,w6]
        N_test_sample, correct_count, test_acc, y_true, y_pred, label_true, n = 600, 0, [ ], [ ], [ ], [ ], 6
        print(self.config.tunedModel_path)
        softmax_func = tf.keras.layers.Softmax( )
        print(f'check for {self.nshots} shots, {N_Rx}_Receivers '
              f'accuracy......................................................................')
        for i in range( N_test_sample ):
            data = self.WidarDataLoaderObj.getMultiDomainSQDataForTest( self.config.nshots, True,
                    Best = self.config.record )
            Support_data, Query_data, Support_embedding, Query_embedding, sim, sim_mean,p = [ ], [ ], [ ], [ ], [ ], \
                                                                                            [ ],[ ]
            keys = list( data[ 'Support_data' ].keys( ) )
            '''gestures, 6 receivers CSI data'''
            # [ Support_data.append( np.concatenate( data[ 'Support_data' ][ keys[ j ] ], axis = 0 ) ) for j in range(len( keys ) ) ]
            # '''gestures, 6 receivers CSI embedding'''
            # for j in range( len( Support_data ) ):
            #     data_buf = Support_data[ j ]
            #     emb_buf = []
            #     for f_idx in range( len( data_buf)):
            #         emb_buf.append(feature_extractor[f_idx].predict( np.expand_dims(data_buf[f_idx],axis=0) ))
            # Support_embedding.append( np.concatenate(emb_buf,axis=0) )
            g_idx = np.random.choice( np.arange( 0, len( keys ) ), 1, replace = False )[ 0 ]
            sample_idx = np.random.choice( np.arange( 0, len(data['Query_data'][keys[g_idx]][0]) ), 1,replace = False)[ 0 ]
            buffer = data['Query_data'][keys[g_idx]]
            ant_selection = np.random.choice( np.arange( 0, 6), N_Rx, replace = False)
            '''select one sample for the selected gesture, corresponding to six receivers'''
            [Query_data.append(buffer[ant][sample_idx]) for ant in range(len(data['Query_data'][keys[g_idx]]))]
            '''select antenna'''
            Query_data_selection = []
            weights_selection = []
            for at in ant_selection:
                Query_data_selection.append(Query_data[at])
                weights_selection.append(weights[at])
            start = time.time( )

            [Query_embedding.append(feature_extractor[ant_selection[ant]].predict(np.expand_dims(
                    Query_data_selection[ant],axis = 0))) for ant in range(len(Query_data_selection))]

            end = time.time( )
            timecost = (end - start) / N_Rx
            print( f'The inference time is: {timecost:.2f}' )
            for w in range(len(Query_embedding)):
                sim.append(cosine_similarity( weights_selection[w], Query_embedding[ w ]) )
            for g in range( len( sim ) ):
                p.append(softmax_func(  np.squeeze(sim[ g ] )).numpy())
            p = np.sum( np.asarray(p), axis = 0 )
            if np.argmax( p ) == g_idx:
                correct_count += 1
        acc = (correct_count / N_test_sample) * 100.
        print( "Accuracy %.2f" % acc )
        return acc
class fineTuningWiAR(fineTuningWidar):
    def __init__( self,config,idx_user):
        super( ).__init__( config = config, isMultiDomain = False,isiheritance = True )
        self.config = config
        self.wiar = WiARdataLoader( config, data_path = f'E:\\Sensing_project\\Cross_dataset\\WiAR\\volunteer_{idx_user}' )
        self.data = self.wiar.data
        self.label = self.wiar.label
        self.config.N_novel_classes = len(self.data)
        self.nways = self.config.N_novel_classes
        self.pretrained_featureExtractor = self._getPreTrainedFeatureExtractor( )
        self.pretrained_featureExtractor.trainable = True
        self.fine_Tune_model = self.modelObj.buildTuneModel(
                pretrained_feature_extractor = self.pretrained_featureExtractor,
                isTest = False, config = self.config
                )
    def tuning( self):
        self.pretrained_featureExtractor.load_weights(self.config.pretrainedfeatureExtractor_path)
        self.pretrained_featureExtractor.trainable = True
        self.data = self.wiar.getSQDataForTest()
        weights = np.transpose(
                self._getNShotsEmbedding( self.pretrained_featureExtractor, self.data[ 'Support_data' ] )
                )
        bias = np.zeros( self.config.N_novel_classes )
        self.fine_Tune_model.get_layer( 'fine_tune_layer' ).set_weights( [ weights, bias ] )
        val_data, val_label = self.data[ 'Val_data' ], to_categorical(
                self.data[ 'Val_label' ], num_classes
                = self.config.N_novel_classes
                )
        optimizer = tf.keras.optimizers.Adam(
                learning_rate = self.config.lr,
                # momentum = 0.9,
                epsilon = 1e-06,
                )
        # optimizer = tf.keras.optimizers.SGD(
        #         learning_rate = config.lr,
        #         momentum = 0.99,
        #         )
        # optimizer = tf.keras.optimizers.Adadelta(
        #         learning_rate = config.lr, rho = 0.50, epsilon = 1e-06, name = 'Adadelta',
        #
        #         )
        # optimizer =  tf.keras.optimizers.RMSprop(
        #                                     learning_rate=config.lr,
        #                                     rho=0.99, momentum=0.9,
        #                                     epsilon=1e-06,
        #                                     centered=False,
        #                                     name='RMSprop',
        #                                                     )
        # optimizer = tf.keras.optimizers.Adamax(
        #                                          learning_rate=self.config.lr, beta_1=0.90, beta_2=0.98, epsilon=1e-08,
        #                                          name='Adamax'
        #                                      )
        self.fine_Tune_model.compile( loss = 'categorical_crossentropy', optimizer = optimizer, metrics = 'acc' )
        idx = np.random.permutation( len( self.data[ 'Support_data' ] ) )
        history = self.fine_Tune_model.fit(
                self.data[ 'Support_data' ][ idx ], to_categorical(self.data[ 'Support_label' ][ idx ] , num_classes
                = self.config.N_novel_classes ),
                epochs = 1000,
                verbose = 0,
                validation_data = (val_data, val_label),
                callbacks = [ self.earlyStop, self.lrScheduler ]
                )
        return [self.fine_Tune_model,self.data['record'],history]
def searchBestSample(nshots=None,rx = None,userId = None):
    config = getConfig( )
    data_dir = [ 'E:/Sensing_project/Cross_dataset/20181109/User1',
                 'E:/Sensing_project/Cross_dataset/20181109/User2',
                 'E:/Sensing_project/Cross_dataset/20181109/User3'
                 ]
    num_sample_per_gesture = {'1':20,'2':20,'3':10}
    x = userId
    config.nshots_per_domain = None
    # config.nshots = int(5*1*1*config.nshots_per_domain)
    config.nshots = nshots
    config.train_dir = data_dir[userId - 1]
    # config.train_dir = 'E:/Cross_dataset/20181115'
    config.N_novel_classes = 6
    config.n_ft_cls = config.N_novel_classes
    config.lr = 1e-4
    config.domain_selection = (2, 2, rx)
    # config.pretrainedfeatureExtractor_path = './models/signFi_featureExtractor_weight_AlexNet_lab_training_acc_0.95_on_250cls.h5'
    config.pretrainedfeatureExtractor_path = 'Code_ds/OneShotGestureRecognition/a.h5'
    fineTuningWidarObj = fineTuningWidar(config = config,isMultiDomain = False)
    location,orientation,Rx = config.domain_selection
    val_acc = 0
    acc_record = []
    results_save_path = 'G:\我的云端硬盘\Widar_results\Widar_resutls.mat'
    # config.tunedModel_path = 'D:\OneShotGestureRecognition\models\Publication_related\widar_fineTuned_model_20181109_1shots__domain(2, 2, 1)_0.36_newFE_user1.h5'
    for i in range(100):
        fine_Tune_model,record,history = fineTuningWidarObj.tuning( init_bias = True,num_sample_per_gesture = num_sample_per_gesture[str(userId)])
        print("=========================================================iteration: "
              "%d=========================================================" % i)
        acc_record.append(history.history[ 'val_acc' ][ -1 ])
        if val_acc < history.history['val_acc'][-1]:
            val_acc = history.history[ 'val_acc' ][ -1 ]
            config.tunedModel_path = f'./models/Publication_related/widar_fineTuned_model_20181109' \
                                     f'_{config.nshots}shots_' \
                                     f'_domain{config.domain_selection}_{val_acc:0.2f}_newFE_user{x}.h5'
            model_to_save = copy.deepcopy(fineTune_model)
            best_record = record
            config.record = best_record
            print(f'Updated record is: {best_record}')
            mdic = {'record':best_record,
                    'val_acc':val_acc}
            # config.setMatSavePath(f"./Sample_index/Publication_related/sample_index_record_for_{config.nshots}_shots"
            #                       f"_domain_{config.domain_selection}_20181109.mat")
            config.setMatSavePath(
                    f"./Sample_index/Publication_related/sample_index_record_for_{config.nshots}_shots"
                    f"_domain_{config.domain_selection}_20181109_newFE_user{x}.mat"
                    )
            savemat( config.matPath, mdic )
            name = f'Domain_{config.domain_selection}_User_{x}_shots_{nshots}'
            if os.path.exists(results_save_path):
                results_mat = loadmat(results_save_path)
            else:
                results_mat = { }
            result_mat = {name:val_acc}
            results_mat.update(result_mat)
            savemat(results_save_path, results_mat)
            if val_acc >= 0.95:
                print(f'reached expected val_acc {val_acc}')
                break
    model_to_save.save_weights( config.tunedModel_path )

    '''Testing'''
    # config.getSampleIdx( )
    # test_acc,[y_true,y_pred],label_true = fineTuningWidarObj.test(applyFinetunedModel = True)
    # plt_cf = pltConfusionMatrix( )
    # title = f'{config.nshots}_shot_sRx_{Rx}_domain_{config.domain_selection}'
    # plt_cf.pltCFMatrix( y = label_true, y_pred = y_pred, figsize = (18,15),title = title )
    # print(f'The average accuracy is {np.mean(acc_record)}')
    # plt.savefig(f'C:/Users/29073/iCloudDrive/PhD Research Files/Publications/One-Shot learning/Results/results_figs'
    #             f'/{config.nshots}shots_'
    #             f'{config.domain_selection}_finetuned_{test_acc[0]:0.2f}_user{x}.pdf')
    return val_acc
def searchBestSampleMultiRx(nshots:int = None,Rx:list = None):
    config = getConfig( )
    x = 3
    data_dir = [ '/media/b218/HOME/b218/Guolin/Code_ds/SensingDataset/Widar/20181109/User1',
                 '/media/b218/HOME/b218/Guolin/Code_ds/SensingDataset/Widar/20181109/User2',
                 '/media/b218/HOME/b218/Guolin/Code_ds/SensingDataset/Widar/20181109/User3'
                 ]
    config.pretrainedfeatureExtractor_path = 'Code_ds/OneShotGestureRecognition/a.h5'
    config.nshots = nshots
    config.train_dir = data_dir[ int(x-1) ]
    config.N_novel_classes = 6
    config.lr = 1e-4
    config.domain_selection = Rx
    # selection = np.random.choice(config.domain_selection,n,replace = False)

    val_acc = 0.0
    fineTuningWidarObj = fineTuningWidar( config = config, isMultiDomain = True )
    for i in range(50):
        fine_Tune_model,record,history = fineTuningWidarObj.tuningMultiRx( )
        print("=========================================================iteration: "
              "%d=========================================================" % i)
        if val_acc < np.max(history.history['val_acc']):
            val_acc = np.max(history.history['val_acc'])
            config.tunedModel_path = f'./models/Publication_related/Rx_specific/widar_fineTuned_M_Rx_model_20181109' \
                                     f'_{config.nshots}shots_' \
                                     f'_domain{config.domain_selection}_{val_acc:0.2f}_newFE_user{x}.h5'
            # fine_Tune_model.save_weights(config.tunedModel_path)
            config.record = record
            print(f'Updated record is: {config.record }')
            mdic = {'record':config.record ,
                    'val_acc':val_acc}
            config.setMatSavePath(
                    f"./Sample_index/Publication_related/sample_index_record_for_{config.nshots}_shots"
                    f"_domain_{config.domain_selection}_20181109_MultiRx_newFE_user{x}.mat"
                    )
            # savemat( config.matPath, mdic )
            if val_acc >= 0.75:
                print(f'reached expected val_acc {val_acc}')
                break
    return val_acc
def evaluationMultiRx(N_Rx,N_shots,userId = 1):
    acc = {'user1':[],'user2':[],'user3':[] }
    config = getConfig( )
    data_dir = [ 'E:/Sensing_project/Cross_dataset/20181109/User1',
                 'E:/Sensing_project/Cross_dataset/20181109/User2',
                 'E:/Sensing_project/Cross_dataset/20181109/User3'
                 ]
    config.pretrainedfeatureExtractor_path = './models/Publication_related/FE/a.h5'
    path = './models/Publication_related/Rx_specific/'

    # config.tunedModel_path = ['./models/Publication_related/widar_fineTuned_M_Rx_model_20181109_1shots__domain['
    #                           '1]_0.37_newFE_user1.h5',
    #                           './models/Publication_related/widar_fineTuned_M_Rx_model_20181109_1shots__domain['
    #                           '2]_0.61_newFE_user1.h5',
    #                           './models/Publication_related/widar_fineTuned_M_Rx_model_20181109_1shots__domain['
    #                           '3]_0.53_newFE_user1.h5',
    #                           './models/Publication_related/widar_fineTuned_M_Rx_model_20181109_1shots__domain['
    #                           '4]_0.39_newFE_user1.h5',
    #                           './models/Publication_related/widar_fineTuned_M_Rx_model_20181109_1shots__domain['
    #                           '5]_0.45_newFE_user1.h5',
    #                           './models/Publication_related/widar_fineTuned_M_Rx_model_20181109_1shots__domain['
    #                           '6]_0.42_newFE_user1.h5',]
    # config.tunedModel_path = './models/Publication_related/widar_fineTuned_M_Rx_model_20181109_1shots__domain[1]_0.37_newFE_user1.h5'
    # config.tunedModel_path = './models/Publication_related/widar_fineTuned_M_Rx_model_20181109_1shots__domain[1, 2]_0.43_newFE_user1'
    # config.tunedModel_path = './models/Publication_related/widar_fineTuned_M_Rx_model_20181109_1shots__domain[1, 2, 3]_0.42_newFE_user1.h5'
    # config.tunedModel_path = './models/Publication_related/widar_fineTuned_M_Rx_model_20181109_1shots__domain[1, 2, 3, 4]_0.41_newFE_user1.h5'
    # config.tunedModel_path = './models/Publication_related/widar_fineTuned_M_Rx_model_20181109_1shots__domain[1, 2, 3, 4, 5]_0.40_newFE_user1.h5'
    config.N_novel_classes = 6
    config.nshots = N_shots
    config.tunedModel_path = [
                            path+f'widar_fineTuned_M_Rx_model_20181109_{config.nshots}shots__domain[1]_newFE_user{userId}.h5',
                            path+f'widar_fineTuned_M_Rx_model_20181109_{config.nshots}shots__domain[2]_newFE_user{userId}.h5',
                            path+f'widar_fineTuned_M_Rx_model_20181109_{config.nshots}shots__domain[3]_newFE_user{userId}.h5',
                            path+f'widar_fineTuned_M_Rx_model_20181109_{config.nshots}shots__domain[4]_newFE_user{userId}.h5',
                            path+f'widar_fineTuned_M_Rx_model_20181109_{config.nshots}shots__domain[5]_newFE_user{userId}.h5',
                            path+f'widar_fineTuned_M_Rx_model_20181109_{config.nshots}shots__domain[6]_newFE_user{userId}.h5',]
    config.matPath = [ f"./Sample_index/Publication_related/sample_index_record_for_{config.nshots}_shots_domain_["
                       f"1]_20181109_MultiRx_newFE_user{userId}",
                       f"./Sample_index/Publication_related/sample_index_record_for_{config.nshots}_shots_domain_["
                       f"2]_20181109_MultiRx_newFE_user{userId}",
                       f"./Sample_index/Publication_related/sample_index_record_for_{config.nshots}_shots_domain_["
                       f"3]_20181109_MultiRx_newFE_user{userId}",
                       f"./Sample_index/Publication_related/sample_index_record_for_{config.nshots}_shots_domain_["
                       f"4]_20181109_MultiRx_newFE_user{userId}",
                       f"./Sample_index/Publication_related/sample_index_record_for_{config.nshots}_shots_domain_["
                       f"5]_20181109_MultiRx_newFE_user{userId}",
                       f"./Sample_index/Publication_related/sample_index_record_for_{config.nshots}_shots_domain_["
                       f"6]_20181109_MultiRx_newFE_user{userId}" ]
    config.record = []
    for i in range(len(config.matPath)):
        config.record.append(loadmat( config.matPath[i] + '.mat' )[ 'record' ])
    # for j in range(3):
    config.train_dir = data_dir[ userId - 1 ]
    config.domain_selection = [1,2,3,4,5,6]
    testWidar = fineTuningWidar( config, True )
    # fineTuneModelEvalObj = WidarDataloader( config = config, isMultiDomain = True )
    # data = fineTuneModelEvalObj.getMultiDomainSQDataForTest(1,False)
    for i in range(10):
        # acc['user1'].append(testWidar.testMultiRx( applyFinetunedModel = True,useWeightMatrix = True,n_Rx = N_Rx ))
        acc[ f'user{userId}' ].append( testWidar.testMultiRxSep( N_Rx = N_Rx ) )
    return acc
def evaluation( domain_selection,nshots ):
    # config.getFineTunedModelPath( )
    # location, orientation, Rx = config.domain_selection
    data_dir = [ 'E:/Sensing_project/Cross_dataset/20181109/User1',
                 'E:/Sensing_project/Cross_dataset/20181109/User2',
                 'E:/Sensing_project/Cross_dataset/20181109/User3'
                 ]
    config = getConfig( )
    config.domain_selection = domain_selection
    config.nshots = nshots
    config.pretrainedfeatureExtractor_path = './models/Publication_related/FE/a.h5'
    # config.setMatSavePath(
    #         f"./Sample_index/Publication_related/sample_index_record_for_2_shots_domain_(2, 2, 3)_20181109.mat"
    #         )
    config.matPath = [f"./Sample_index/sample_index_record_for_{nshots}_shots_domain_[1]_20181109_MultiRx_newFE_user1",
                      f"./Sample_index/sample_index_record_for_{nshots}_shots_domain_[2]_20181109_MultiRx_newFE_user1"
                      f"./Sample_index/sample_index_record_for_{nshots}_shots_domain_[3]_20181109_MultiRx_newFE_user1"
                      f"./Sample_index/sample_index_record_for_{nshots}_shots_domain_[4]_20181109_MultiRx_newFE_user1"
                      f"./Sample_index/sample_index_record_for_{nshots}_shots_domain_[5]_20181109_MultiRx_newFE_user1"
                      f"./Sample_index/sample_index_record_for_{nshots}_shots_domain_[6]_20181109_MultiRx_newFE_user1"]
    # config.getSampleIdx( )
    config.train_dir = data_dir[0]
    # config.tunedModel_path = f'./models/fine_tuning_widar/widar_fineTuned_model_20181109_1shots_test_domain_(2, 2, 3).h5'
    config.tunedModel_path = './models/Publication_related/widar_fineTuned_model_20181109_5shots__domain(2, 2, ' \
                             '3)_0.97_newFE_user2.h5'
    config.record = loadmat(config.matPath)['record']
    config.domain_selection = domain_selection
    config.N_novel_classes = 6
    fineTuneModelEvalObj = fineTuningWidar( config = config, isMultiDomain = False )
    test_acc,[y_true,y_pred],label_true = fineTuneModelEvalObj.test(applyFinetunedModel =True)
    plt_cf = pltConfusionMatrix( )
    plt_cf.pltCFMatrix(
            y = label_true, y_pred = y_pred, figsize = (12, 10),title = ""
            )
    return test_acc
def compareDomain():
    config = getConfig( )
    config.nshots_per_domain = 2
    config.nshots = int( 5 * 1 * 1 * config.nshots_per_domain )
    config.train_dir = 'E:/Cross_dataset/20181109/User1'
    config.N_novel_classes = 6
    config.lr = 1e-3
    config.domain_selection = (2, 2, 3)
    config.pretrainedfeatureExtractor_path = \
        './models/feature_extractor_weight_Alexnet_lab_250cls_val_acc_0.996_no_zscore.h5'
    config.tunedModel_path = \
        f'./models/MultiDomain_Widar' \
        f'/widar_fineTuned_model_20181109_10shots_MultiDomainOrientation_Rx3_location_2_multi_csiANP_.h5'
    fineTuningWidarObj = fineTuningWidar( config = config, isMultiDomain = True )
    feature_extractor, classifier = fineTuningWidarObj._loadFineTunedModel(
            applyFinetunedModel = True
            )
    feature_extractor = fineTuningWidarObj._getoutput( feature_extractor )

    WidarDataLoaderObj223 = WidarDataloader(
            dataDir = config.train_dir, selection = (2,2,3), isMultiDomain = False,
            config = config
            )
    data223 = WidarDataLoaderObj223.getSQDataForTest(
            nshots = 1, mode = 'fix',
            isTest = False
            )
    WidarDataLoaderObj233 = WidarDataloader(
            dataDir = config.train_dir, selection = (3, 5, 1), isMultiDomain = False,
            config = config
            )
    data233 = WidarDataLoaderObj233.getSQDataForTest(
            nshots = 1, mode = 'fix',
            isTest = False
            )
    pred_223 = feature_extractor.predict(data223['Val_data'])
    label_223 = data223['Val_label']
    pred_233 = feature_extractor.predict( data233['Val_data'] )
    label_233 = data233[ 'Val_label' ]
    class_t_sne( pred_223, label_223,perplexity = 40, n_iter = 3000 )
def tuningSignFi(preTrain_model_path,nshots,source = 'home',n_ft_cls=None):
    config = getConfig( )
    config.source = source
    config.nshots = nshots
    # config.n_ft_cls = n_ft_cls
    # config.N_novel_classes = 25
    # config.N_base_classes = 276 - config.N_novel_classes

    config.N_novel_classes = n_ft_cls
    config.N_base_classes = 276 - config.N_novel_classes
    # config.lr = 4e-4
    # config.lr = 1e-3
    # user 1 - 0.7e-3, 2,3 - 0.65e-3, 4 -
    # config.lr = 1e-3
    config.lr = 6.5e-4
    # config.lr = 3e-4
    config.train_dir = 'D:\Matlab\SignFi\Dataset'
    # config.tunedModel_path = f'./models/Publication_related/signFi_finetuned_model_{config.nshots}_shots_' \
    #                          f'{config.N_novel_classes}_ways_256_1280.h5'
    # config.pretrainedfeatureExtractor_path = 'a.h5'
    config.pretrainedfeatureExtractor_path = preTrain_model_path
    tuningSignFiObj = fineTuningSignFi(config,isZscore = False)
    fine_Tune_model = tuningSignFiObj.tuning(init_weights = True,init_bias = False)
    return fine_Tune_model,config
def testingSignFi(path,mode,N_train_classes,environment:str):
    config = getConfig( )
    config.nshots = 1
    config.train_dir = 'D:\Matlab\SignFi\Dataset'
    config.source = 'home'
    config.N_base_classes = 200
    config.N_novel_classes = 76
    all_path = os.listdir( f'./models/pretrained_feature_extractors/' )

    config.tunedModel_path = path
    tuningSignFiObj = fineTuningSignFi( config, isZscore = False )
    acc_all = tuningSignFiObj.test( nway = None, applyFinetunedModel = True )

    return acc_all
def tuningWiar(nshots,idx_user):
    config = getConfig()
    config.nshots = nshots
    config.lr = 1e-4
    config.pretrainedfeatureExtractor_path = './models/Publication_related/FE/a.h5'
    wiarFTObj = fineTuningWiAR(config = config,idx_user=idx_user)
    val_acc = 0.0
    for i in range(50):
        fine_Tune_model,record,history = wiarFTObj.tuning()
        print("=========================================================iteration: "
              "%d=========================================================" % i)
        print('Tuning %d shots model' % nshots)
        currentAcc = np.max( history.history[ 'val_acc' ])
        print(f'Accuracy is {currentAcc}')
        if val_acc < np.max(history.history['val_acc']):
            val_acc = np.max(history.history['val_acc'])
            config.tunedModel_path = f'.\\models\\Publication_related\\wiar_FT_{config.nshots}shots_' \
                                     f'{val_acc:0.2f}_User_{idx_user}.h5'
            fine_Tune_model.save_weights(config.tunedModel_path)
            print(f'Updated record is: {record}')
            mdic = {'record':record,
                    'val_acc':val_acc}
            config.setMatSavePath(
                    f".\\Sample_index\\Publication_related\\sample_index_record_for_{config.nshots}_shots_User_{idx_user}.mat"
                    )
            savemat( config.matPath, mdic )
    return val_acc
if __name__ == '__main__':
    '''Widar tuning and testing'''
    # shots_con = [2,3,4,5]
    # rx_container = [1,2,3,4,5,6]
    # userId_container = [1,2,3]
    # for shot in shots_con:
    #     for rx in rx_container:
    #         for userId in userId_container:
    #             re = loadmat('G:\我的云端硬盘\Widar_results\Widar_resutls.mat', squeeze_me = 1)
    #             domain_selection = (2, 2, rx)
    #             name_check = f'Domain_{domain_selection}_User_{userId}_shots_{shot}'
    #             if name_check in list(re.keys()):
    #                 continue
    #             searchBestSample(nshots=shot,rx = rx,userId = userId)
    if 0:
        preTrain_model_path = f'a.h5'
        # for n_ft_cls in np.linspace(1,50,13,dtype = int):
        for shot in [5]:
            results_path = f'compare_TF_{shot}Shot.mat'
            if os.path.exists( results_path ):
                acc_all = loadmat( results_path )
            else:
                acc_all = { }
            for n_ft_cls in [76]:
                # if str(n_ft_cls) in acc_all.keys():
                #     continue
                fine_Tune_model, config = tuningSignFi( preTrain_model_path, shot, 'home', n_ft_cls)
                acc = np.max( config.history.history[ 'val_acc' ] )
                model_name = f'signFi_home_FT_{n_ft_cls}_classes_ACC_{ acc}_direct_compare_{shot}_shots.h5'
                save_dir = os.path.join( 'models/Publication_related/Transfer_learning_comparing/same_number_of_novel_classes', model_name )
                fine_Tune_model.save( save_dir )

                acc = { str(n_ft_cls) : np.max( config.history.history[ 'val_acc' ] ) }

                acc_all.update( acc )
                savemat(results_path, acc_all)
                print(acc_all)

    if 0:
        # acc_all = {
        #         'user_5':[],
        #         'user_4':[],
        #         'user_3':[],
        #         'user_2':[],
        #         'user_1':[],
        #         'user_e':[],
        #         }
        # # N_base_classes = 170
        acc_all = { }
        preTrain_model_path = f'a.h5'
        path_all = []
        for path in os.listdir('D:\OneShotGestureRecognition\models\pretrained_feature_extractors'):
            if 'signFi_featureExtractor_weight_AlexNet_lab_training' in path and 'acc' not in path and 'FT' not in path:
                path_all.append(path)
        for path in path_all:
            FT_model_path = path.split('.')[0]+'_targetDomain_lab'+'.'+path.split('.')[1]
            path = os.path.join('D:\OneShotGestureRecognition\models\pretrained_feature_extractors', path)
            FT_model_path = os.path.join('D:\OneShotGestureRecognition\models', FT_model_path)
            fine_Tune_model,config = tuningSignFi(path,1,'lab')
            fine_Tune_model.save(FT_model_path)
            acc= {path.split('\\')[-1].split('.')[0]:np.max(config.history.history['val_acc'])}
            acc_all.update(acc)
            print(acc_all)
            # acc_all.append(np.max(config.history.history['val_acc']))

    # for N_base_classes in np.linspace(3,20,18,dtype = int):
        # for N_base_classes in [30,40,50,60,70,80,90,100,110,120,150,200,]:
        #     preTrain_model_path = f'D:\OneShotGestureRecognition\models\pretrained_feature_extractors\signFi_featureExtractor_weight_AlexNet_lab_training_{N_base_classes}cls.h5'
        #     FT_model_path = f'D:\OneShotGestureRecognition\models\pretrained_feature_extractors\FT_a_lab_training_{N_base_classes}cls.h5'
        # for i in ['home',[1,2,3,4,5]]
        # user_N = [[4,3,1,2,5],[4,1,2,5,3],[4,3,1,5,2],[3,1,2,5,4,]]
        #     user_N = ['lab',]
        # for _,i in enumerate([1,2,3,4,5]):
        #     for n in user_N:
        #     fine_Tune_model,config = tuningSignFi(preTrain_model_path,1,'lab')
        #     acc_all[f'user_{n[-1]}'].append(np.max(config.history.history['val_acc']))
        # fe = Model( inputs = fine_Tune_model.input, outputs = fine_Tune_model.get_layer( 'lambda_layer' ).output)
        # fine_Tune_model.save_weights(FT_model_path)
        # acc_all = np.asarray( acc_all )
    ''' 25 classes
    home: 1 -> 88.03
    home: 2 -> 84.5 93.5
    home: 3 -> 86.5
    home: 4 -> 92.5
    home: 5 -> 93.6
    lab 2:[0.68000001, 0.72000003, 0.78857142, 0.81999999, 0.83999997] - user 5
    lab 2:[0.92888892, 0.97000003, 0.96571428, 0.97333336, 0.96799999] - user 4
    lab 2:[0.80000001, 0.86000001, 0.88      , 0.89333332, 0.912     ] - user 3
    lab 2:[0.68000001, 0.70499998, 0.71428573, 0.75333333, 0.792     ] - user 2
    average: [0.77222224, 0.81375001, 0.83714286, 0.86      , 0.87799999]
    
    6 novel classes
    'home': [1,1,1,1,1]
    'user_5': [0.26, 0.29, 0.52, 0.58, 0.60],
    'user_4': [1.0, 1.0, 1.0, 1.0, 1.0],
    'user_3': [1.0, 1.0, 1.0, 1.0, 1.0],
    # 'user_2': [0.93,0.85,0.86,0.92,0.97]
    # 'user_2': [0.85, 0.92, 0.79, 0.88, 0.87]
    'user_2': [0.85, 0.86, ,0.89, 0.92, 0.97]

    '''
    # preTrain_model_path = f'a.h5'
    
    # fine_Tune_model, config = tuningSignFi( preTrain_model_path, 1, 'home' )
    # savemat( f'signfi_oneshot_history.mat', config.history.history )
    # testingSignFi('fe.h5','Alexnet',200,'home')
    for i in range(1,6):
        for j in range(1,7):
            searchBestSampleMultiRx(nshots = 5,Rx = [j])
    # tuningSignFi()
    '''WiAR'''
    # acc_all = { }
    # shots_list = [1,2,3,4,5]
    # for nshots in shots_list:
    #     val_acc = tuningWiar(nshots = nshots,idx_user = 6 )
    #     acc_all[f'User1_{nshots}_shots'] = val_acc
    # key = list(acc_all.keys())
    # for i in range(len(key)):
    #     print(acc_all[key[i]])
    '''Multiple receivers test'''
    # acc_all = {}
    # userId = 3
    # for i in range(1,7):
    #     for shot in range(1,6):
    #         acc = evaluationMultiRx( N_Rx = i,N_shots=shot,userId = userId )
    #         acc_all[f'{i}_Rx_{shot}_shot'] = acc
    # savemat(f'multiRx_acc_user_{userId}.mat',acc_all)
    # for k in range(1,7):
    #     for x in range(1,6):
    #         per = np.mean(acc_all[f'{k}_Rx_{x}_shot']['user3'])
    #         print(f'{k} receivers, {x} shots accuracy is: {per:.2f}%')