import numpy as np
from PIL import Image
import tensorly as tl
import scipy.io as sio
import skvideo.io
from os import walk

# TODO(fahrbach): Include input and output paths for each tensor.

def get_output_filename_prefix(input_filename):
    tokens = input_filename.split('/')
    assert(tokens[0] == 'data')
    tokens[0] = 'output'
    output_filename_prefix = '/'.join(tokens)
    return output_filename_prefix.split('.')[0]

class TensorDataHandler:
    def __init__(self):
        self.input_filename = None
        self.tensor = None
        self.output_path = None

    def generate_random_tucker(self, shape, rank, random_state=1234):
        self.tensor = tl.random.random_tucker(shape, rank, full=True,
                random_state=random_state)
        self.output_path = 'output/random_tucker/'
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
        self.output_path = get_output_filename_prefix(input_filename)
        return self.tensor

    def load_cardiac_mri(self):
        """
        shape: (256, 256, 14, 20)
        size: 18,350,080
        """
        self.input_filename = 'data/cardiac-mri/sol_yxzt_pat1.mat'
        self.tensor = sio.loadmat(self.input_filename)['sol_yxzt'].astype(float)
        self.output_path = 'output/cardiac-mri'
        return self.tensor

    def load_hyperspectral(self):
        """
        shape: (1024, 1344, 33)
        size: 45,416,488
        https://personalpages.manchester.ac.uk/staff/d.h.foster/Time-Lapse_HSIs/nogueiro/nogueiro_1140.zip
        """
        self.input_filename = 'data/hyperspectral/nogueiro_1140.mat'
        self.tensor = sio.loadmat(self.input_filename)['hsi'].astype(float)
        self.output_path = 'output/hyperspectral'
        return self.tensor

    def load_coil_100(self):
        """
        shape: (7200, 120, 120, 3)
        size: 311,040,000
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
