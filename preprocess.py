import os
import glob
import argparse
import numpy as np

import cv2

from helper import MHIProcessor

def count_frames(path):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total

def split_indexes(video_files):
    np.random.seed(42)
    idxs = np.arange(len(video_files))
    np.random.shuffle(idxs)
    counts = [count_frames(video_files[idx]) for idx in idxs]
    splits = np.array([0.7, 0.9])
    train, val = tuple(splits * sum(counts))
    
    tt_count = 0
    for idx, count in enumerate(counts):
        tt_count += count
        if tt_count >= train:
            idx1 = idx + 1
            break
    for idx, count in enumerate(counts[idx1:]):
        tt_count += count
        if tt_count >= val:
            idx2 = idx1 + idx + 1
            break
    return idx1, idx2, idxs

def fall_annotations(video_files):
    falls = []
    for file in video_files:
        file = file.replace('Videos', 'Annotation_files')
        file = file.replace('.avi', '.txt')
        with open(file) as f:
            lines = f.readlines()
            falls.append((int(lines[0]), int(lines[1]))) # (start, stop)
    return falls

def prepare_train_val_test(location, src_dir):
    video_files = glob.glob(f'{src_dir}/{location}/Videos/*')
    video_files = sorted(video_files)
    train_end_idx, val_end_idx, idxs = split_indexes(video_files)
    fall_idxs = fall_annotations(video_files)
    train = [(video_files[idx], fall_idxs[idx]) for idx in idxs[:train_end_idx]]
    val = [(video_files[idx], fall_idxs[idx]) for idx in idxs[train_end_idx:val_end_idx]]
    test = [(video_files[idx], fall_idxs[idx]) for idx in idxs[val_end_idx:]]
    return train, val, test

def create_MHI(data, dst, dataset='train'):
    dest_path = f'{dst}/{dataset}'

    os.makedirs(f'{dest_path}/fall', exist_ok=True)
    os.makedirs(f'{dest_path}/not_fall', exist_ok=True)

    for file_path, annotation in data: 
        prefix = file_path.split('/')
        prefix = f'{prefix[1]}_{prefix[-1][7:9]}'
        prefix = prefix.replace(')', '')  # remove ')' for one-digit number i.e. 3)
        start_fall, stop_fall = annotation

        cap = cv2.VideoCapture(file_path)

        mhi_processor = MHIProcessor()

        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret : break

            img = mhi_processor.process(frame, save_batch=True)
            frame_id = mhi_processor.index

            if img is not None:
                if frame_id >= start_fall and frame_id <= stop_fall:
                    cv2.imwrite(f'{dest_path}/fall/{prefix}_{frame_id}.png',img)
                else:
                    cv2.imwrite(f'{dest_path}/not_fall/{prefix}_{frame_id}.png',img)

        cap.release()

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument('-s', '--source', required=True, default='data',
                    help='Source data folder')
    ap.add_argument('-d', '--dest', required=True, default='dataset',
                    help='Destination dataset folder')

    args = vars(ap.parse_args())

    src_dir = args['source']

    locations = os.listdir(src_dir)
    
    for location in locations:
        print(f'Start {location}')
        train, val, test = prepare_train_val_test(location, src_dir)
        datasets = {'train': train,
                    'val': val,
                    'test': test}
        for key, value in datasets.items():
            create_MHI(value, dst=args['dest'], dataset=key)

    print('Finish Preprocessing')

        
if __name__ == "__main__":
    main()