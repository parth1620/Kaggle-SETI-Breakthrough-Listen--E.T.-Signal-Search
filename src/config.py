params = {
    'DATA_DIR' : '/content/train/',
    'TRAIN_CSV' : '/content/train_folds (6).csv',
    'MODEL_PATH' : '/content/drive/MyDrive/SETI_MODELS/',

    'SEED' : 42,
    'FOLD' : 4,
    'IMG_SIZE' : 512,

    'EPOCHS' : 20,
    'BATCH_SIZE' : 32,

    'NUM_WORKERS' : 4,
    'DEVICE' : 'cuda',
    'OPTIMIZER' : 'ranger',
    'CRIT' : 'bce',
    'SCHEDULER' : 'cosine',

    'OUTPUT_DIM' : 1,
    'MODEL_NAME' : 'eca_nfnet_l0',
    'DEVICE' : 'cuda',

    'LR' : 0.0005,
    'T_MAX' : 20,
    'ETA_MIN' : 1e-6,

    'SAM': False,
    'SWA' : False,
    'FP16' : True,
    'SWA_START' : 35,

    'SCHEDULER_PARAMS' : {
        "lr_start": 1e-4,
        "lr_max": 1e-4 * 32,
        "lr_min": 1e-6,
        "lr_ramp_ep": 6,
        "lr_sus_ep": 0,
        "lr_decay": 0.75,
    }
}
