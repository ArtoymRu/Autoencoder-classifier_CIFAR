from keras.datasets import cifar10, cifar100
import os
import requests
import tarfile
import zipfile


def load_dataset(dataset_name):
    '''
    Load standart CIFAR10/CIFAR100 datasets into working directory
    Arguments:
        dataset_name: str
    Return:
        train/test split of dataset
        (x_train, y_train), (x_test, y_test)
    '''
    if dataset_name == 'CIFAR10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        # Normalize the images
        x_train, x_test = x_train.astype('float32')/255.0, x_test.astype('float32')/255.0
        return (x_train, y_train), (x_test, y_test)
    
    elif dataset_name == 'CIFAR100':
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        # Normalize the images
        x_train, x_test = x_train.astype('float32')/255.0, x_test.astype('float32')/255.0
        return (x_train, y_train), (x_test, y_test)
    

def download_dataset(url):
    '''
    Download dataset from the link to the data folder
    Arguments:
        url: str
    Return:
        None
    '''
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(parent_dir, 'data')

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    filename = url.split('/')[-1]

    if os.path.exists(os.path.join(data_dir, filename)):
        print(f'{filename} already exists in the data folder. Skipping download.')
        return

    print(f'Downloading {filename}...')
    response = requests.get(url, stream=True)
    with open(os.path.join(data_dir, filename), 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    print(f'{filename} downloaded successfully.')

    if filename.endswith('.tar.gz'):
        print(f'Extracting {filename}...')
        with tarfile.open(os.path.join(data_dir, filename), 'r:gz') as tar:
            tar.extractall(data_dir)
        print(f'{filename} extracted successfully.')
    elif filename.endswith('.zip'):
        print(f'Extracting {filename}...')
        with zipfile.ZipFile(os.path.join(data_dir, filename), 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print(f'{filename} extracted successfully.')