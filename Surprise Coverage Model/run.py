import numpy as np
import time
import argparse
import numpy as np

import gzip
from tqdm import tqdm
from keras.datasets import mnist, cifar10
from keras.models import load_model, Model
from sa import fetch_dsa, fetch_lsa, get_sc
from utils import *
import os
import glob
import cv2
CLIP_MIN = -0.5
CLIP_MAX = 0.5

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="mnist")
    parser.add_argument(
        "--lsa", "-lsa", help="Likelihood-based Surprise Adequacy", action="store_true"
    )
    parser.add_argument(
        "--dsa", "-dsa", help="Distance-based Surprise Adequacy", action="store_true"
    )
    # dsa for classification
    parser.add_argument(
        "--target",
        "-target",
        help="Target input set (test or adversarial set)",
        type=str,
        default="fgsm",
    )
    parser.add_argument(
        "--save_path", "-save_path", help="Save path", type=str, default="./tmp/"
    )
    parser.add_argument(
        "--batch_size", "-batch_size", help="Batch size", type=int, default=128
    )
    parser.add_argument(
        "--var_threshold",
        "-var_threshold",
        help="Variance threshold",
        type=int,
        default=1e-5,
    )
    parser.add_argument(
        "--upper_bound", "-upper_bound", help="Upper bound", type=int, default=2000
    )
    parser.add_argument(
        "--n_bucket",
        "-n_bucket",
        help="The number of buckets for coverage",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--num_classes",
        "-num_classes",
        help="The number of classes",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--is_classification",
        "-is_classification",
        help="Is classification task",
        type=bool,
        default=True,
    )
    args = parser.parse_args()
    assert args.d in ["mnist", "cifar"], "Dataset should be either 'mnist' or 'cifar'"
    assert args.lsa ^ args.dsa, "Select either 'lsa' or 'dsa'"
    print(args)
    os.makedirs(args.save_path, exist_ok=True)
    if args.d == "mnist":
        
        path = 'original_data/'
        files = ['train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz'
                 ,'t10k-images-idx3-ubyte.gz','t10k-labels-idx1-ubyte.gz']
        paths = [path+each for each in files]
        with gzip.open(paths[1],'rb') as imgpath:
            y_train = np.frombuffer(imgpath.read(),np.uint8,offset=8)
        with gzip.open(paths[0],'rb') as imgpath:
            x_train = np.frombuffer(imgpath.read(),np.uint8,offset=16).reshape(len(y_train),28,28,1)
        with gzip.open(paths[3],'rb') as imgpath:
            y_test= np.frombuffer(imgpath.read(),np.uint8,offset=8)
        with gzip.open(paths[2],'rb') as imgpath:
            x_test = np.frombuffer(imgpath.read(),np.uint8,offset=16).reshape(len(y_test),28,28,1)
        
        # Load pre-trained model.
        model = load_model("./model/model_{}.h5".format(args.d))             #model_mnist.h5
        model.summary()

        # You can select some layers you want to test.
        # layer_names = ["activation_1"]
        # layer_names = ["activation_2"]
        # layer_names = ["activation_1", "activation_3"]
        # layer_names = ["activation_4"]
        layer_names = ["activation_1"]

        # Load target set.
        X_data = []
        files = glob.glob ('/home/armin/Desktop/artifacts_eval/adv_samples/{}/{}/*.png'.format(args.d, args.target))
        for myFile in files:
            image = cv2.imread (myFile, cv2.IMREAD_GRAYSCALE)
            X_data.append (image)

        # print('X_data shape:', np.array(X_data).shape)
        x_target = np.array(X_data).astype('float32')
        x_target = (x_target / 255) - (1.0 - CLIP_MAX)
        x_target = np.expand_dims(x_target, axis=3)
        # x_target = np.load("./adv/adv_mnist_{}.npy".format(args.target))

    elif args.d == "cifar":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        model = load_model("./model/model_cifar.h5")
        model.summary()

        # layer_names = [
        #     layer.name
        #     for layer in model.layers
        #     if ("activation" in layer.name or "pool" in layer.name)
        #     and "activation_9" not in layer.name
        # ]
        layer_names = ["activation_6"]

        # x_target = np.load("./adv/adv_cifar_{}.npy".format(args.target))

        # print('X_data shape:', np.array(X_data).shape)
        # x_target = np.array(X_data).astype('float32')

        X_data = []
        files = glob.glob ('/home/armin/Desktop/artifacts_eval/adv_samples/{}/{}/*.png'.format(args.d, args.target))
        for myFile in files:
            image = cv2.imread (myFile)
            X_data.append (image)

        # print('X_data shape:', np.array(X_data).shape)
        x_target = np.array(X_data).astype('float32')

    x_train = x_train.astype("float32")
    x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
    x_test = x_test.astype("float32")
    x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)

    if args.lsa:
        if os.path.exists('./adv/adv_ltest.npy'):
            test_lsa = np.load("./adv/adv_ltest.npy")
        
        else:
            test_lsa = fetch_lsa(model, x_train, x_test, "test", layer_names, args)
            np.save('./adv/adv_ltest.npy',test_lsa)
        target_lsa = fetch_lsa(model, x_train, x_target, args.target, layer_names, args)
        target_cov = get_sc(
            np.amin(target_lsa), args.upper_bound, args.n_bucket, target_lsa
        )

        auc = compute_roc_auc(test_lsa, target_lsa)
        print(infog("ROC-AUC: " + str(auc * 100)))

    if args.dsa:
        # SAMPLE_SIZE = 1 * 300
        # x_train = x_train[:SAMPLE_SIZE]
        # x_test = x_test[:SAMPLE_SIZE]
        # x_target = x_target[:SAMPLE_SIZE]
        if os.path.exists('./adv/adv_{}_test.npy'.format(args.d)):# and False:
            test_dsa = np.load('./adv/adv_{}_test.npy'.format(args.d))
        
        else:
            test_dsa = fetch_dsa(model, x_train, x_test, "test", layer_names, args)
            np.save('./adv/adv_{}_test.npy'.format(args.d),test_dsa)
        target_dsa = fetch_dsa(model, x_train, x_target, args.target, layer_names, args)
        target_cov = get_sc(
            np.amin(target_dsa), args.upper_bound, args.n_bucket, target_dsa
        )
        print(len(target_dsa), len(test_dsa))
        auc = compute_roc_auc(test_dsa, target_dsa)
        print(infog("ROC-AUC: " + str(auc * 100)))

    print(infog("{} coverage: ".format(args.target) + str(target_cov)))
    f=open('./result/sa_conv5_{}_dsa.txt'.format(args.d),'a+')
    result = '{} {} {} {}'.format(args.target, target_cov, auc, layer_names)
    f.write(result)
    f.write('\n')
    f.close()
