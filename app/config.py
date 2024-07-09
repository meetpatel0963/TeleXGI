class Config:
    SEED = 2024

    # Model Parameters
    RESOLUTION = 224
    M_MLP_INPUT_DIM = 1280
    R_MLP_INPUT_DIM = 2048
    MLP_HIDDEN_DIM = 512
    MLP_DROPOUT_RATE = 0.2

    NUM_CLASSES = 8

    RESNET50_MODEL_PATH = '../models/resnet50/model_fold_0_epoch_30.pth'
    MOBILENETV2_MODEL_PATH = '../models/mobilenetv2/model_fold_0_epoch_30.pth'

    # Models Available
    RESNET50 = 'resnet50'
    MOBILENETV2 = 'mobilenetv2'
    MODEL_WEIGHTS = {
        RESNET50: RESNET50_MODEL_PATH, 
        MOBILENETV2: MOBILENETV2_MODEL_PATH, 
    }
        
    # Model Used
    MODEL_NAME = RESNET50

    # Dataset
    CLASS_NAMES = ['dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', \
                   'normal-cecum', 'normal-pylorus', 'normal-z-line', 'polyps', 'ulcerative-colitis']
    MEAN = [0.4857, 0.3460, 0.2983]
    STD_DEV = [0.3348, 0.2456, 0.2369]

    # Sample Data
    BASE_PATH = '../data'
    SAMPLE_IMAGES_PATH = BASE_PATH + '/images'
    SAMPLE_VIDEOS_PATH = BASE_PATH + '/videos'

    # Evaulation
    CONFIDENCE_THRESHOLD = 0.999
