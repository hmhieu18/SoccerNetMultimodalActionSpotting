# import ffmpeg
import traceback
import time
import glob
import os

import numpy as np
import tensorflow as tf

from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from preprocess_sound import preprocess_sound
from scipy.io import wavfile
from vggish import VGGish

# ROOT_PATH = r'D:\DeepSport\Video'
ROOT_PATH = r'T:\SoccerNet\data'
OUT_DIR_PATH = r'D:\DeepSport\SoccerNet-code\data'


def getShapeWithoutLoading(numpyFile):
    with open(numpyFile, 'rb') as f:
        major, minor = np.lib.format.read_magic(f)
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
    return shape


def create_directory(dir_name):
    if not os.path.exists(dir_name) or not os.path.isdir(dir_name):
        os.mkdir(dir_name)


def containsFile(root, filename):
    return len([os.path.join(root, f) for f in os.listdir(root)
                if filesAreEqual(root, f, filename)]) > 0


def filesAreEqual(root, f1, f2):
    fPath = os.path.join(root, f1)
    return os.path.isfile(fPath) and fPath.lower() == f2.lower()


def readFile(filename):
    f = open(filename)
    lines = f.readlines()
    lines = [l.replace('\n', '') for l in lines]
    f.close()
    return lines


def removeFile(filename):
    if os.path.exists(filename):
        os.remove(filename)


def subdirectories(root):
    return [os.path.join(root, d) for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))]


def generate_newname(filepath, postfix, extension):
    basename = os.path.splitext(filepath)[0]
    new_name = basename + postfix + "." + extension
    return new_name


def extractDirFeatures(gameDir, extractor, outDir=None):
    if outDir == None:
        outDir = gameDir
    else:
        outDir = gameDir.replace(ROOT_PATH, outDir)

    starts = startTimes(gameDir)

    extractHalfTimeFeatures(gameDir, outDir, starts[0], extractor, '1')
    extractHalfTimeFeatures(gameDir, outDir, starts[1], extractor, '2')

    # removeFile(audioPath)

    # featuresFilePath = os.path.join(outDir, '2_VGGish.npy')
    # if not containsFile(gameDir, featuresFilePath):
    #     audioPath = convert2wav(gameDir, '2_HQ', outDir=outDir)
    #     features = extractFeatures(gameDir, audioPath, starts[1], extractor)
    #     np.save(featuresFilePath, features)
    #     removeFile(audioPath)


def extractHalfTimeFeatures(gameDir, outDir, startTime, extractor, half):
    featuresFilePath = os.path.join(outDir, half + '_VGGish.npy')
    if not containsFile(outDir, featuresFilePath):
        audioPath = convert2wav(gameDir, half + '_HQ', outDir=outDir)
        videoFeaturesPath = os.path.join(outDir, half + '_ResNET_PCA512.npy')
        videoFeatures = np.load(videoFeaturesPath)
        features = extractFeatures(
            gameDir, audioPath, videoFeatures.shape[0], startTime, extractor)
        np.save(featuresFilePath, features)
        removeFile(audioPath)


def startTimes(directory):
    videoIniFilePath = os.path.join(gameDir, 'video.ini')
    videoIniContent = readFile(videoIniFilePath)
    starts = [0, 0]
    starts[0] = int(videoIniContent[1].split()[-1])
    starts[1] = int(videoIniContent[4].split()[-1])

    return starts


def extractFeatures(gameDir, audioFilePath, numVideoFeatures, startTime, extractor):
    sr, wav_data = wavfile.read(audioFilePath)
    deltaTime = int(0.5*sr)
    # Remove start of video
    wav_data = wav_data[startTime * sr:]
    # Select 45 min of game
    if numVideoFeatures == 5400:
        wav_data = wav_data[:45*60*sr]

    seg_num = 60
    total_seg = int(len(wav_data) / deltaTime)

    start = 0
    cur_seg = 0
    features = np.zeros((total_seg, 512))
    while cur_seg < total_seg:
        data = np.zeros((seg_num, 46, 64, 1))
        for i in range(seg_num):
            if start + deltaTime > len(wav_data):
                break
            cur_wav = wav_data[start:start + deltaTime]
            cur_wav = cur_wav / 32768.0
            cur_spectro = preprocess_sound(cur_wav, sr)
            cur_spectro = np.expand_dims(cur_spectro, 3)
            data[i, :, :, :] = cur_spectro
            start += deltaTime
            cur_seg += 1
        current_features = extractor.predict(data)
        features[cur_seg-current_features.shape[0]                 :cur_seg, :] = current_features

    if features.shape[0] > numVideoFeatures:
        features = features[:numVideoFeatures, :]
    return features


def extractSound(gameDir, outDir=None):
    convert2wav(gameDir, '1_HQ', outDir=outDir)
    convert2wav(gameDir, '2_HQ', outDir=outDir)


def convert2wav(gameDir, file_suffix, outDir):
    audioPath = os.path.join(outDir, file_suffix + '.wav')
    videoPath = os.path.join(gameDir, file_suffix + '.mkv')
    ffmpeg.input(videoPath).output(audioPath).run()

    return audioPath


def tensorflowInit():
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    # config = tf.ConfigProto(gpu_options=gpu_options)
    # config.gpu_options.allow_growth = True
    # session = tf.Session(config=config)
    import tensorflow as tf
    from tensorflow.compat.v1.keras.backend import set_session

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True
    sess = tf.compat.v1.Session(config=config)
    set_session(sess)
    vgg_base = VGGish(include_top=False, load_weights=True)
    x = vgg_base.get_layer(name='conv4/conv4_2').output
    output_layer = GlobalAveragePooling2D()(x)
    model = Model(vgg_base.inputs, output_layer)

    return model

# if __name__ == '__main__':
#     model = tensorflowInit()

#     for competitionDir in subdirectories(ROOT_PATH):
#         # if competitionDir != "D:\\DeepSport\\SoccerNet-code\\data\\spain_laliga":
#         #     continue
#         create_directory(competitionDir.replace(ROOT_PATH, OUT_DIR_PATH))
#         for yearDir in subdirectories(competitionDir):
#             # if competitionDir != "D:\\DeepSport\\SoccerNet-code\\data\\spain_laliga\\2014-2015":
#             #     continue
#             create_directory(yearDir.replace(ROOT_PATH, OUT_DIR_PATH))
#             for gameDir in subdirectories(yearDir):
#                 print(gameDir)
#                 # if gameDir == "T:\\SoccerNet\\data\\spain_laliga\\2014-2015\\2015-02-14 - 20-00 Real Madrid 2 - 0 Dep. La Coruna":
#                 create_directory(gameDir.replace(ROOT_PATH, OUT_DIR_PATH))
#                 extractDirFeatures(gameDir, model, outDir=OUT_DIR_PATH)


model = tensorflowInit()

dir_name = '/content/drive/MyDrive/Thesis_temp/soccernet-video'
video_path = ""
# Get list of all files only in the given directory
list_of_files = filter(os.path.isfile,
                       glob.glob(dir_name + '/**/*.wav', recursive=True))
# Sort list of files based on last modification time in ascending order
list_of_files = sorted(list_of_files,
                       key=os.path.getmtime)
# Iterate over sorted list of files and print file path
# along with last modification time of file
convertedFilePath = os.path.join(dir_name, "vggish-converted-files-list.txt")
if os.path.isfile(convertedFilePath):
    f = open(convertedFilePath, "r")
    convertedFiles = f.read()
else:
    convertedFiles = []
print("LENGTH", len(list_of_files))
for file_path in list_of_files:
    if(not file_path in convertedFiles):
        # timestamp_str = time.strftime(  '%m/%d/%Y :: %H:%M:%S',
        #                             time.gmtime(os.path.getmtime(file_path)))
        print(file_path, "is being converted")
        try:
            # wavName = main(file_path)
            # extractFeatures(file_path)
            # extractDirFeatures(file_path, model)
            basename = os.path.basename(file_path)

            gameName = file_path.split(basename)[0]
            gameName = gameName.split(dir_name)[-1]

            visualFilepath = os.path.join(
                video_path, gameName, basename[0]+"_ResNET_TF2.npy")

            featuresFilePath = generate_newname(file_path, '_VGGish', 'npy')
            # if not containsFile(gameDir, featuresFilePath):
            print("visualFilepath", visualFilepath)

            shape = getShapeWithoutLoading(visualFilepath)
            features = extractFeatures(file_path, file_path, 0, model)
            
            np.save(featuresFilePath, features)
            with open(convertedFilePath, "a") as file_object:
                # Append 'hello' at the end of file
                file_object.write(file_path + '\n')
        except Exception:
            traceback.print_exc()
        # Open a file with access mode 'a'

    else:
        print(file_path, "is converted, skip")
