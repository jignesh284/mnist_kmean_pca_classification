import numpy as np
import datetime
import sys
import os
from sklearn.decomposition import PCA
from scipy.spatial import distance



SET_SIZE = 800
OUTPUT_FILE = './2350611767.txt'

configs = {
    'IMAGES': 'train-images-idx3-ubyte',
    'LABELS': 'train-labels-idx1-ubyte',
    'TYPE_IMAGE': 'image',
    'TYPE_LABEL': 'label',
}

class KNN():
    def __init__(self, N, PATH_TO_DATA_DIR):
        self.PATH_TO_DATA_DIR = PATH_TO_DATA_DIR
        images = self.convert_ubyte_array(os.path.join(PATH_TO_DATA_DIR, configs['IMAGES']), configs['TYPE_IMAGE'])[0:SET_SIZE, :]
        labels = self.convert_ubyte_array(os.path.join(PATH_TO_DATA_DIR, configs['LABELS']), configs['TYPE_LABEL'])[0:SET_SIZE, :]
        self.test_images = images[:N, :]
        self.train_images = images[N:, :]
        self.train_labels = labels[N:, :] 
        self.test_labels = labels[:N, :]


    def convert_ubyte_array(self, filename, file_type):
        """return numpy array after pasing the input mnist images/labels ubyte files"""
        with open(filename, 'rb') as f:
            magic_num = self._readbyte(f)
            num_item = self._readbyte(f)
            width = 1
            if file_type == configs['TYPE_IMAGE']:
                rows = self._readbyte(f)
                cols = self._readbyte(f)
                width = rows * cols
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape((num_item, width))
            return data

    @staticmethod
    def _readbyte(file):
        """ reads the 4 bytes from file pointer and convert it integer """
        return int.from_bytes(file.read(4), 'big')
    
    @staticmethod
    def l2_distance(train_images, test_images):
        """ calculates l2 norm of a test point with rest of the training points """
        return np.sqrt(np.sum((train_images - test_images)**2, axis=1))

    
    def write_output(self, predictions):
        """ writes the result in the output file """
        text = ''
        for i in range(len(self.test_labels)):
            text = text + f'{predictions[i]} {self.test_labels[i][0]}\n'

        with open(OUTPUT_FILE, 'w') as f:
            f.write(text)
    
    def reduce_dimension(self, D):
        """ Reduces the input dimensions to D using PCA """
        pca = PCA(n_components=D, svd_solver='full')
        pca.fit(self.train_images)
        test_images = pca.transform(self.test_images)
        train_images = pca.transform(self.train_images)
        return (train_images, test_images)

    def predict(self, train_images, test_images, K):
        """ Predicts the labels for test images for KNN """
        predictions = []
        for i in range(len(test_images)):
            test_img = test_images[i]
            dist = self.l2_distance(train_images, test_img)
            sorted_dist = dist[dist.argsort()]
            sorted_labels = self.train_labels[dist.argsort()]
            weighted_freq = np.zeros(10)
            for j in range(K):
                weighted_freq[sorted_labels[j][0]] += (1/sorted_dist[j])

            predictions.append(np.argmax(weighted_freq, axis=0))
        return predictions
    
    def accuracy(self, predictions):
        """ gives the precertage of accuracy in predictions """
        correct = 0
        total = len(self.test_labels)
        for i in range(len(self.test_labels)):
            if predictions[i] == self.test_labels[i][0]:
                correct += 1 
        
        return (correct / total)


if __name__ == "__main__":
    time1 = datetime.datetime.now()
    try:
        K = int(sys.argv[1])
        D = int(sys.argv[2])
        N = int(sys.argv[3])
        PATH_TO_DATA_DIR = sys.argv[4]
    except:
        print("Exception occured while parsing parameters")

    knn = KNN(N, PATH_TO_DATA_DIR)

    reduced_train_images, reduced_test_images = knn.reduce_dimension(D)

    predictions = knn.predict(reduced_train_images, reduced_test_images, K)

    knn.write_output(predictions)

    # print(f'Accuracy :: {knn.accuracy(predictions)}')
    time2 = datetime.datetime.now()

    elapsedTime = time2 - time1
    print(f'Elapsed Time :: {elapsedTime.total_seconds()}')

    

