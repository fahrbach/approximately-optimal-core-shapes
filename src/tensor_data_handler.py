import numpy as np
import csv
from PIL import Image
import tensorly as tl
import scipy.io as sio
import skvideo.io
from os import walk
import sparse
from tensorly.contrib.sparse import tensor as stensor
from tensorly.contrib.sparse import unfold as sunfold

# TODO(fahrbach): Include input and output paths for each tensor.

def get_output_filename_prefix(input_filename):
    tokens = input_filename.split('/')
    assert(tokens[0] == 'data')
    tokens[0] = 'output'
    output_filename_prefix = '/'.join(tokens)
    return output_filename_prefix.split('.')[0]

# Rename to TensorDataLoader
class TensorDataHandler:
    def __init__(self):
        self.input_filename = None
        self.tensor = None
        self.output_path = None

    def make_random_tensor(self, shape, seed=0):
        np.random.seed(seed)
        self.tensor = np.random.random_sample(shape)
        self.output_path = 'output/random-tensor_'
        self.output_path += 'shape-' + '-'.join([str(x) for x in shape]) + '_'
        self.output_path += 'seed-' + str(seed) + '/'
        return self.tensor

    def generate_random_tucker(self, shape, rank, random_state=1234):
        self.tensor = tl.random.random_tucker(shape, rank, full=True,
                random_state=random_state)
        self.output_path = 'output/random-tucker_'
        self.output_path += 'shape-' + '-'.join([str(x) for x in shape]) + '_'
        self.output_path += 'rank-' + '-'.join([str(x) for x in rank]) + '_'
        self.output_path += 'seed-' + str(random_state) + '/'
        return self.tensor

    def generate_synthetic_shape(self, pattern='swiss', image_height=20,
            image_width=20, n_channels=None):
         self.tensor = tl.datasets.synthetic.gen_image(pattern, image_height,
                 image_width, n_channels)
         self.output_path = 'output/synthetic_shapes/' + pattern
         return self.tensor

    def load_image(self, input_filename, resize_shape=None):
        self.input_filename = input_filename
        image = Image.open(input_filename)
        if resize_shape:
            image = image.resize((resize_shape[0], resize_shape[1]), Image.ANTIALIAS)
        self.tensor = np.array(image) / 256

        self.output_path = 'output/'
        self.output_path += input_filename.split('/')[-1].split('.')[0]
        self.output_path += '_shape-' + '-'.join([str(x) for x in self.tensor.shape])
        self.output_path += '/'
        return self.tensor

    def load_cardiac_mri(self):
        """
        shape: (256, 256, 14, 20)
        size: 18,350,080
        """
        self.input_filename = 'data/cardiac-mri/sol_yxzt_pat1.mat'
        self.tensor = sio.loadmat(self.input_filename)['sol_yxzt'].astype(float)
        self.output_path = 'output/cardiac-mri/'
        return self.tensor

    def load_hyperspectral(self):
        """
        shape: (1024, 1344, 33)
        size: 45,416,488
        https://personalpages.manchester.ac.uk/staff/d.h.foster/Time-Lapse_HSIs/nogueiro/nogueiro_1140.zip
        """
        self.input_filename = 'data/hyperspectral/nogueiro_1140.mat'
        self.tensor = sio.loadmat(self.input_filename)['hsi'].astype(float)
        self.output_path = 'output/hyperspectral/'
        return self.tensor

    def load_hands(self):
        """
        shape: (60, 80, 30, 900)
        size: 129,600,000
        https://labicvl.github.io/ges_db.htm
        """
        assert False

    def load_traffic(self):
        """
        shape: (1084, 2033, 96)
        size: 211,562,112

        Paper: "Traffic forecasting in complex urban networks: Leveraging big data and machine learning"
        Source: https://github.com/florinsch/BigTrafficData
        """

        self.input_filename = 'data/traffic/VolumeData_tensor.mat'
        self.tensor = sio.loadmat(self.input_filename)['data'].astype(float)
        self.output_path = 'output/traffic/'
        return self.tensor

    def load_coil_100(self):
        """
        shape: (7200, 128, 128, 3)
        size: 353,894,400
        """
        path = 'data/coil-100'
        image_files = []
        for (dirpath, dirnames, filenames) in walk(path):
            for filename in filenames:
                filename = filename.strip()
                if filename[-4:] != '.png': continue
                image_files.append(dirpath + '/' + filename)
        image_files = sorted(image_files)
        assert len(image_files) == 7200
        images = []
        for filename in image_files:
            image = np.array(Image.open(filename))
            images.append(image)
        self.tensor = np.array(images) / 256.0
        self.output_path = 'output/coil-100/'
        return self.tensor

    # Video in Tucker-TensorSketch paper:
    # https://github.com/OsmanMalik/tucker-tensorsketch
    def load_video(self, input_filename):
        self.input_filename = input_filename
        print('loading video:', input_filename)
        self.tensor = skvideo.io.vread(input_filename)
        print('video shape:', self.tensor.shape)
        self.output_path = get_output_filename_prefix(input_filename)
        return self.tensor
        
        
    # https://archive.ics.uci.edu/ml/datasets/Mushroom
    def load_mushroom(self):
        """
        shape: (6, 4, 10, 2, 9, 2, 2, 2, 12, 2, 5, 4, 4, 9, 9, 1, 4, 3, 5, 9, 6, 7)
        array_size: (8124, 23)
        label: binary in first column
        """
        self.input_filename = 'data/mushroom/mushroom.csv'
        with open(self.input_filename) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            data = np.zeros((8124, 23))
            counter = 0
            for row in reader:
                data[counter, :] = row
                counter += 1

        shape = [0] * (data.shape[1] - 1)
        for i in range(len(shape)):
            shape[i] = len(set(data[:, i+1]))
        coords = (data[:, 1:] - 1).astype(int)
        vals = (data[:, 0] - 1.5) * 2
        X = sparse.COO(coords.T, vals, shape=tuple(shape))
        self.tensor = stensor(X, dtype='float')
        return self.tensor
