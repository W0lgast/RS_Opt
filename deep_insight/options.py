"""

Defines options dict for training.

"""

import deep_insight.loss as loss

class Param_Holder:
    def __init__(self):
        self.empty = True

GLOBALS = Param_Holder()

def make_globals(args):

    GLOBALS.rat_name = args.name
    GLOBALS.similarity_penalty = args.simpen
    GLOBALS.train_half_key = args.trainkey
    GLOBALS.epochs = args.epochs

    #GLOBALS.model_path = f"models/{GLOBALS.rat_name}-Train-{GLOBALS.train_half_key}-SimPen-{GLOBALS.similarity_penalty}-epoch-{GLOBALS.epochs}.pt"
    GLOBALS.model_path = "models/Herman-Train-mj_top-SimPen-1000000-epoch-5000.pt"
    #GLOBALS.model_path = "models/Felix-Train-None-SimPen-10000-epoch-5000.pt"
    GLOBALS.mat_path = f"data/{GLOBALS.rat_name}.mat"
    GLOBALS.h5_path = f"data/{GLOBALS.rat_name}.h5"

    # TARGETS = ["position", "head_direction", "direction", "speed"] #Felix, Gerrit, Etc...
    GLOBALS.targets = ["position", "speed", "direction"]

    loss_functions = {'position': 'euclidean_loss',
                      'speed': 'mse', #was mae
                      'head_direction': 'cyclical_mae_rad',
                      'direction': 'cyclical_mae_rad',
                      'direction_delta': 'cyclical_mae_rad'}
    GLOBALS.loss_functions = {t: getattr(loss, loss_functions[t]) for t in GLOBALS.targets}

    if GLOBALS.rat_name in ["POSTSKIP", "PRESKIP"]:
        loss_weights = {'position': 20,
                        'speed': 400,
                        'head_direction': 25,
                        'direction': 2.5,
                        'direction_delta': 25,
                        }
    elif GLOBALS.rat_name in ["Elliott", "Felix", "Gerrit", "Herman", "Ibsen"]:
        loss_weights = {'position': 1,
                        'speed': 20,
                        'head_direction': 25,
                        'direction': 200,
                        'direction_delta': 25,
                        }
    else:
        print("ERROR: Unknown rat name - what weights should I use?")
        exit(0)
    GLOBALS.loss_weights = {t: loss_weights[t] for t in GLOBALS.targets}

    GLOBALS.empty = False

def get_globals():
    if GLOBALS.empty:
        print("ERROR: No global vars have been made!")
        exit(0)
    return GLOBALS

def get_opts(fp_hdf_out, train_test_times):
    """
    Returns the options dictionary which contains all parameters needed to train the model

    ..todo:: which of these opts are redundant? take them out.
    """
    opts = dict()

    # -------- DATA ------------------------
    opts['fp_hdf_out'] = fp_hdf_out  # Filepath for hdf5 file storing wavelets and outputs
    opts['sampling_rate'] = 1250 # 512*4 # Sampling rate of the wavelets
    opts['training_indices'] = train_test_times[0].tolist()  # Indices into wavelets used for training the model, adjusted during CV
    opts['testing_indices'] = train_test_times[1].tolist()  # Indices into wavelets used for testing the model, adjusted during CV
    #opts['channels'] = 16
    opts['channels'] = None

    # -------- MODEL PARAMETERS --------------
    opts['model_function'] = 'Standard_Decoder'  # Model architecture used
    opts['model_timesteps'] = 64
    # 32 #64  # How many timesteps are used in the input layer, e.g. a sampling rate of 30 will yield 2.13s windows. Has to be divisible X times by 2. X='num_convs_tsr'
    opts['num_convs_tsr'] = 5  # Number of downsampling steps within the model, e.g. with model_timesteps=64, it will downsample 64->32->16->8->4 and output 4 timesteps
    opts['average_output'] = 2**opts['num_convs_tsr']  # Whats the ratio between input and output shape

    opts['optimizer'] = 'adam'  # Learning algorithm
    opts['learning_rate'] = 0.0007  # Learning rate
    opts['kernel_size'] = 3  # Kernel size for all convolutional layers
    opts['conv_padding'] = 'same'  # Which padding should be used for the convolutional layers
    opts['act_conv'] = 'ELU'  # Activation function for convolutional layers
    opts['act_fc'] = 'ELU'  # Activation function for fully connected layers
    opts['dropout_ratio'] = 0  # Dropout ratio for fully connected layers
    opts['filter_size'] = 64  # Number of filters in convolutional layers
    opts['num_units_dense'] = 1024  # Number of units in fully connected layer
    opts['num_dense'] = 2  # Number of fully connected layers
    opts['gaussian_noise'] = 1  # How much gaussian noise is added (unit = standard deviation)

    # -------- TRAINING----------------------
    opts['batch_size'] = 8  # Batch size used for training the model
    opts['steps_per_epoch'] = 250  # Number of steps per training epoch
    opts['validation_steps'] = 15  # Number of steps per validation epoch #..todo: val happens once per epoch now, this var is redundant
    opts['epochs'] = get_globals().epochs  # Number of epochs
    opts['shuffle'] = False  # If input should be shuffled
    opts['random_batches'] = True  # If random batches in time are used
    opts['num_cvs'] = 3 # the number of cross validation splits

    # -------- MISC--------------- ------------
    opts['tensorboard_logfolder'] = './'  # Logfolder for tensorboard
    opts['model_folder'] = './'  # Folder for saving the model
    opts['log_output'] = False  # If output should be logged
    opts['save_model'] = False  # If model should be saved

    return opts