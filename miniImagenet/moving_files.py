import os
import shutil
import glob
import argparse

parser = argparse.ArgumentParser(
    description="Moving unlabel images in one folder."
    )

parser.add_argument('-r', '--RootDir', type=str, default=r'./dataset/train/', help='Path to root dataset')
parser.add_argument('-t', '--DestFolder', type=str, default=r'./dataset/unsupervised/', help='Path to target folder')
parser.add_argument('-n', '--num', type=int, default=600, help='Number of images per class to move')

# Parse the arguments
args = parser.parse_args()
print(args)

# move selected number of images
if not os.path.exists(args.DestFolder):
    os.makedirs(args.DestFolder)
for root, dirs, files in os.walk((os.path.normpath(args.RootDir)), topdown=False):
        for i in dirs:
            SourceFolder = os.path.join(args.RootDir,i)
            for filename in glob.glob(os.path.join(SourceFolder, '*.*'))[:args.num]: # select number of files
                shutil.copy(filename, args.DestFolder)
