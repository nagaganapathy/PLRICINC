#!/usr/bin/env python

# Edit this script to add your team's training code.

from helper_code import *
import numpy as np, os, sys, joblib
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from scipy import ndimage
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold
import csv
import pandas as pd
from numpy.lib import stride_tricks
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import transforms
import tensorflow as tf
import cv2

################################################################################
#
# Training function
#
################################################################################

# Train your model. This function is **required**. Do **not** change the arguments of this function.
def training_code(data_directory, model_directory):
    # Find header and recording files.
    print('Finding header and recording files...')

    header_files, recording_files = find_challenge_files(data_directory)
    num_recordings = len(recording_files)

    # Create a folder for the model if it does not already exist.
    if not os.path.isdir(model_directory):
        os.mkdir(model_directory)

    # Extract classes from dataset.
    print('Extracting classes...')

    classes = set()
    for header_file in header_files:
        header = load_header(header_file)
        classes |= set(get_labels(header))
        #print(classes)
    if all(is_integer(x) for x in classes):
        classes = sorted(classes, key=lambda x: int(x))
    else:
        classes = sorted(classes)
    num_classes = len(classes)
    print('Extracting classes...###################')

    # Extract features and labels from dataset.
    print('Extracting features and labels...')

    data = np.zeros((num_recordings, 14), dtype=np.float32) # 14 features: one feature for each lead, one feature for age, and one feature for sex
    labels = np.zeros((num_recordings, num_classes), dtype=np.bool) # One-hot encoding of classes

    os.remove("/content/drive/MyDrive/python-classifier-2021/myfile.csv")

    data_map = {}
    for i in range(10):#range(num_recordings):#range(num_recordings):
        print('    {}/{}...'.format(i+1, num_recordings))

        # Load header and recording.
        header = load_header(header_files[i])
        recording = load_recording(recording_files[i])
        age, sex, rms = get_features(header, recording, twelve_leads)
        data[i, 0:12] = rms        

        current_labels = get_labels(header)
        for label in current_labels:
            if label in classes:
                  train_data = get_signal_spectrum(header, recording, twelve_leads, data_directory,label,i)     
                  j = classes.index(label)  
            #labels[i, j] = 1

    #Train 12-lead ECG model.
    print('Training 12-lead ECG model...')

    leads = twelve_leads
    filename = os.path.join(model_directory, 'twelve_lead_ecg_model.sav')
    col_Names=["filepath", "label"]
    train_data = pd.read_csv('/content/drive/MyDrive/python-classifier-2021/myfile.csv',names=col_Names)
    #channel112 = (ch1, ch2, ch3,ch4,ch5,ch6,ch7, ch8, ch9,ch10,ch11,ch12)
    train_data_chl12 = train_data[train_data['filepath'].str.contains('chl_1|chl_2|chl_3|chl_4|chl_5|chl_6|chl_7|chl_8|chl_9|chl_10|chl_11|chl_12')] 
    classes, leads, classifier = our_clsf_model(data_directory,train_data_chl12,leads)
    feature_indices = [twelve_leads.index(lead) for lead in leads]
    features = data[:, feature_indices]
    imputer=SimpleImputer().fit(features)
    save_model(filename, classes, leads, imputer, classifier)

    # Train 6-lead ECG model.
    print('Training 6-lead ECG model...')

    leads = six_leads
    filename = os.path.join(model_directory, 'six_lead_ecg_model.sav')
    col_Names=["filepath", "label"]
    train_data = pd.read_csv('/content/drive/MyDrive/python-classifier-2021/myfile.csv',names=col_Names)
    #channel6 = (ch1, ch2, ch3,ch4,ch5,ch6)
    train_data_chl6 = train_data[train_data['filepath'].str.contains('chl_1|chl_2|chl_3|chl_4|chl_5|chl_6')] 
    classes, leads, classifier = our_clsf_model(data_directory,train_data_chl6,leads)
    feature_indices = [twelve_leads.index(lead) for lead in leads]
    features = data[:, feature_indices]
    imputer=SimpleImputer().fit(features)
    save_model(filename, classes, leads, imputer, classifier)

    # Train 3-lead ECG model.
    print('Training 3-lead ECG model...')

    leads = three_leads
    filename = os.path.join(model_directory, 'three_lead_ecg_model.sav')
    col_Names=["filepath", "label"]
    train_data = pd.read_csv('/content/drive/MyDrive/python-classifier-2021/myfile.csv',names=col_Names)
    #channel3 = (ch1, ch2, ch8)
    train_data_chl3 = train_data[train_data['filepath'].str.contains('chl_1|chl_2|chl_8')]
    classes, leads, classifier = our_clsf_model(data_directory,train_data_chl3,leads)

    feature_indices = [twelve_leads.index(lead) for lead in leads]
    features = data[:, feature_indices]
    imputer=SimpleImputer().fit(features)
    save_model(filename, classes, leads, imputer, classifier)

    # Train 2-lead ECG model.
    print('Training 2-lead ECG model...')

    leads = two_leads
    filename = os.path.join(model_directory, 'two_lead_ecg_model.sav')

    col_Names=["filepath", "label"]
    train_data = pd.read_csv('/content/drive/MyDrive/python-classifier-2021/myfile.csv',names=col_Names)
    #channel2 = (ch1, ch11)
    train_data_chl2 = train_data[train_data['filepath'].str.contains('chl_1|chl_11')]
    classes, leads, classifier = our_clsf_model(data_directory,train_data_chl2,leads)
    feature_indices = [twelve_leads.index(lead) for lead in leads]
    features = data[:, feature_indices]
    imputer=SimpleImputer().fit(features)
    save_model(filename, classes, leads, imputer, classifier)

################################################################################
#
# File I/O functions
#
################################################################################

# Save your trained models.
def save_model(filename, classes, leads, imputer, classifier):
    # Construct a data structure for the model and save it.
    d = {'classes': classes, 'leads': leads, 'imputer': imputer, 'classifier': classifier}
    joblib.dump(d, filename, protocol=0)

# Load your trained 12-lead ECG model. This function is **required**. Do **not** change the arguments of this function.
def load_twelve_lead_model(model_directory):
    filename = os.path.join(model_directory, 'twelve_lead_ecg_model.sav')
    return load_model(filename)

# Load your trained 6-lead ECG model. This function is **required**. Do **not** change the arguments of this function.
def load_six_lead_model(model_directory):
    filename = os.path.join(model_directory, 'six_lead_ecg_model.sav')
    return load_model(filename)

# Load your trained 2-lead ECG model. This function is **required**. Do **not** change the arguments of this function.
def load_two_lead_model(model_directory):
    filename = os.path.join(model_directory, 'two_lead_ecg_model.sav')
    return load_model(filename)

# Generic function for loading a model.
def load_model(filename):
    return joblib.load(filename)

################################################################################
#
# Running trained model functions
#
################################################################################

# Run your trained 12-lead ECG model. This function is **required**. Do **not** change the arguments of this function.
def run_twelve_lead_model(model, header, recording):
    return run_model(model, header, recording)

# Run your trained 6-lead ECG model. This function is **required**. Do **not** change the arguments of this function.
def run_six_lead_model(model, header, recording):
    return run_model(model, header, recording)

# Run your trained 2-lead ECG model. This function is **required**. Do **not** change the arguments of this function.
def run_two_lead_model(model, header, recording):
    return run_model(model, header, recording)

# Generic function for running a trained model.
def run_model(model, header, recording):
    classes = model['classes']
    leads = model['leads']
    imputer = model['imputer']
    classifier = model['classifier']

    # Load features.
    num_leads = len(leads)
    data = np.zeros(num_leads+2, dtype=np.float32)
    age, sex, rms = get_features(header, recording, leads)
    data[0:num_leads] = rms
    data[num_leads] = age
    data[num_leads+1] = sex

    # Impute missing data.
    features = data.reshape(1, -1)
    features = imputer.transform(features)

    # Predict labels and probabilities.
    labels = classifier.predict(features)
    labels = np.asarray(labels, dtype=np.int)[0]

    probabilities = classifier.predict_proba(features)
    probabilities = np.asarray(probabilities, dtype=np.float32)[:, 0, 1]

    return classes, labels, probabilities

################################################################################
#
# Other functions
#
################################################################################

# Extract features from the header and recording.
def get_features(header, recording, leads):
    # Extract age.
    age = get_age(header)
    if age is None:
        age = float('nan')

    # Extract sex. Encode as 0 for female, 1 for male, and NaN for other.
    sex = get_sex(header)
    if sex in ('Female', 'female', 'F', 'f'):
        sex = 0
    elif sex in ('Male', 'male', 'M', 'm'):
        sex = 1
    else:
        sex = float('nan')

    # Reorder/reselect leads in recordings.
    available_leads = get_leads(header)
    indices = list()
    for lead in leads:
        i = available_leads.index(lead)
        indices.append(i)
    recording = recording[indices, :]

    # Pre-process recordings.
    amplitudes = get_amplitudes(header, leads)
    baselines = get_baselines(header, leads)
    num_leads = len(leads)
    for i in range(num_leads):
        recording[i, :] = amplitudes[i] * recording[i, :] - baselines[i]

    # Compute the root mean square of each ECG lead signal.
    rms = np.zeros(num_leads, dtype=np.float32)
    for i in range(num_leads):
        x = recording[i, :]
        rms[i] = np.sqrt(np.sum(x**2) / np.size(x))

    return age, sex, rms

def get_signal_spectrum(header, recording, leads,data_directory, label, k):
    
    # print('Storing classes...')
    mapped_scored_labels = np.array(list(csv.reader(open('/content/drive/MyDrive/python-classifier-2021/dx_mapping_scored.csv'))))
    data_directory_image = '/content/drive/MyDrive/python-classifier-2021/Database_Image/'
    if not os.path.isdir(data_directory_image):
        os.mkdir(data_directory_image)

    # Reorder/reselect leads in recordings.
    available_leads = get_leads(header)
    indices = list()
    for lead in leads:
        i = available_leads.index(lead)
        indices.append(i)
    recording = recording[indices, :]

    # Pre-process recordings.
    amplitudes = get_amplitudes(header, leads)
    baselines = get_baselines(header, leads)
    num_leads = len(leads)
    for i in range(num_leads):
        recording[i, :] = amplitudes[i] * recording[i, :] - baselines[i]
    
    #img = np.zeros((num_leads,2), dtype=np.float32)
    index_value = np.where(mapped_scored_labels == label)[0]
    for i in range(num_leads):
        if index_value: 
            index_lbl = ''.join(filter(str.isalpha,str(mapped_scored_labels[index_value,2])))
            label_img_path = (data_directory_image +"chl_"+str(i+1)+"/"+ ''.join(filter(str.isalpha,str(mapped_scored_labels[index_value,2])))+"/")
            if not os.path.isdir(label_img_path):
                #os.mkdir(label_img_path)
                os.makedirs(label_img_path, exist_ok = True)
        else:
            index_lbl = str('NR')
            label_img_path = data_directory_image+"chl_"+str(i+1)+"/"+ "NR/"
            if not os.path.isdir(label_img_path):
                #os.mkdir(label_img_path)
                os.makedirs(label_img_path, exist_ok = True)
        x = recording[i, :]
        sample_rate = get_frequency(header)
        samples =  x
        Time = np.linspace(0, len(samples) / sample_rate, num=len(samples))
        filename = str(k+1) +'_'+str(i+1)+'.png'
        plotpath1 = str(label_img_path + filename)
        with open('/content/drive/MyDrive/python-classifier-2021/myfile.csv', 'a') as f: 
          writer = csv.writer(f)
          writer.writerow([str(plotpath1), index_lbl])

        img = plotstft(samples,sample_rate, plotpath=plotpath1)

    return img,index_lbl


""" short time fourier transform of signal """
def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)    
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1

    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize),
                                      strides=(samples.strides[0]*hopSize,
                                      samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)    

""" scale frequency axis logarithmically """    
def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = (np.unique(np.round(scale)))

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            newspec[:,i] = np.sum(spec[:,int(scale[i]):], axis=1)
        else:        
            newspec[:,i] = np.sum(spec[:,int(scale[i]):int(scale[i+1])], axis=1)

    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[int(scale[i]):])]
        else:
            freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i+1])])]

    return newspec, freqs

""" plot spectrogram"""
def plotstft(samples,samplerate, binsize=2**10, plotpath=None, colormap="jet"):

    s = stft(samples, binsize)
    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)
    ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel
    timebins, freqbins = np.shape(ims)
    ims = np.transpose(ims)
    img = cv2.resize(ims, (256, 256))
    img = np.stack((img,)*3, axis=-1).astype(np.float32())
    cv2.imwrite(plotpath, img)
    #img = cv2.applyColorMap(ims, cv2.COLORMAP_JET);

    return img
  

def build_model(num_classes,IMG_SIZE):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    NUM_CLASSES = num_classes
    img_augmentation = Sequential(
    [
        preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
        preprocessing.RandomFlip(),
        preprocessing.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
    )

    x = img_augmentation(inputs)
    #x = inputs
    model = EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = True

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.4
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay( initial_learning_rate=1e-2,
    decay_steps=100,
    decay_rate=0.01)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def our_clsf_model(data_directory,train_data_sp,leads):
    train_data = train_data_sp
    leads = leads
    Y = train_data[['label']] 
    skf = StratifiedKFold(n_splits = 2, random_state = 7, shuffle = True) 
    idg = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.3,fill_mode='nearest',horizontal_flip = True,rescale=1./255)

    data_directory_image = data_directory
    b = "Database"
    image_dir = data_directory_image.replace(b,"Database_Image")
    
    NUM_CLASSES=len(train_data['label'].unique().tolist())
    classes1 = train_data['label'].unique().tolist()
    IMG_SIZE = 224

    VALIDATION_ACCURACY = []
    VALIDATION_LOSS = []

    fold_var = 1
    
    filepath="/content/drive/MyDrive/python-classifier-2021/Model/weights_best.hdf5"

    for train_index, val_index in skf.split(np.zeros(len(Y)),Y):
    		training_data = train_data.iloc[train_index]
    		validation_data = train_data.iloc[val_index]
    	
    		train_data_generator = idg.flow_from_dataframe(training_data, directory = image_dir, x_col = "filepath", y_col = "label", target_size=(IMG_SIZE , IMG_SIZE ),
    							   class_mode = "categorical", shuffle = True)
    		valid_data_generator  = idg.flow_from_dataframe(validation_data, directory = image_dir,	x_col = "filepath", y_col = "label", target_size=(IMG_SIZE , IMG_SIZE ),
    							class_mode = "categorical", shuffle = True)
 
        #imputer=SimpleImputer().fit(train_data_generator)
    		classifier = build_model(num_classes=NUM_CLASSES,IMG_SIZE = IMG_SIZE )
    
      
            # CREATE CALLBACKS
    		checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1,save_best_only=True, mode='max')
    		callbacks_list = [checkpoint]

    		#model.summary()
        
    		hist = classifier.fit(train_data_generator, epochs=5, callbacks=callbacks_list, validation_data=valid_data_generator, verbose=1)
            
      		# LOAD BEST MODEL to evaluate the performance of the model
    		classifier.load_weights(filepath)
    	
    		results = classifier.evaluate(valid_data_generator)
    		results = dict(zip(classifier.metrics_names,results))
    	
    		VALIDATION_ACCURACY.append(results['accuracy'])
    		VALIDATION_LOSS.append(results['loss'])
      
    		valid_data_generator.reset()
    		y_pred = classifier.predict_generator(valid_data_generator,verbose=1)
    		y_pred_bool = np.argmax(y_pred, axis=1)
      
    		print(confusion_matrix(valid_data_generator.classes,y_pred_bool))
    		print (y_pred_bool)

    		labels = (valid_data_generator.class_indices)
    		labels = dict((v,k) for k,v in labels.items())
    		predictions = [labels[k] for k in y_pred_bool]

    		filenames=valid_data_generator.filenames
    		results=pd.DataFrame({"Filename":filenames,"Predictions":predictions})
    		results.to_csv("results.csv",index=False)   

    		tf.keras.backend.clear_session()
    	
    		fold_var += 1
    return classes1, leads, classifier 
