from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os.path as osp
import glob
import re
import os
from ..dataset import ImageDataset

# Sources :
# https://github.com/hh23333/PVPM
# https://github.com/lightas/Occluded-DukeMTMC-Dataset
# Miao, J., Wu, Y., Liu, P., DIng, Y., & Yang, Y. (2019). "Pose-guided feature alignment for occluded person re-identification". ICCV 2019

class OccludedDuke_Aug(ImageDataset):
    """OccludedDuke with augmentation support.
    
    Similar structure to Market1501_Aug, supporting augmented images from 
    aug_duke_inpainting_5prompts directory.
    """
    _junk_pids = [0, -1]
    dataset_dir = 'Occluded_Duke'
    masks_base_dir = 'masks'
    cam_num = 8
    train_dir = 'bounding_box_train'
    query_dir = 'query'
    gallery_dir = 'bounding_box_test'
    pattern = re.compile(r'([-\d]+)_c(\d)')

    masks_dirs = {
        # dir_name: (parts_num, masks_stack_size, contains_background_mask)
        'pifpaf': (36, False, '.jpg.confidence_fields.npy'),
        'bpbreid_masks': (8, True, '.npy'),
        'pifpaf_maskrcnn_filtering': (36, False, '.jpg.confidence_fields.npy'),
        'isp_6_parts': (5, True, '.jpg.confidence_fields.npy', ["p{}".format(p) for p in range(1, 5+1)])
    }

    @staticmethod
    def get_masks_config(masks_dir):
        if masks_dir not in OccludedDuke_Aug.masks_dirs:
            return None
        else:
            return OccludedDuke_Aug.masks_dirs[masks_dir]

    def __init__(self, root='', masks_dir=None, **kwargs):
        self.kp_dir = kwargs['config'].model.kpr.keypoints.kp_dir
        self.masks_dir = masks_dir
        if self.masks_dir in self.masks_dirs:
            self.masks_parts_numbers, self.has_background, self.masks_suffix = self.masks_dirs[self.masks_dir]
        else:
            self.masks_parts_numbers, self.has_background, self.masks_suffix = None, None, None
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        # self.train_dir = osp.join(self.dataset_dir, self.train_dir)
        # aug 넣으려고 - Market1501_Aug와 동일하게 변경
        self.train_dir = self.dataset_dir
        self.query_dir = osp.join(self.dataset_dir, self.query_dir)
        self.gallery_dir = osp.join(self.dataset_dir, self.gallery_dir)

        required_files = [
            self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)

        super(OccludedDuke_Aug, self).__init__(train, query, gallery, **kwargs)
        
        # 새로 추가한 부분 - Market1501_Aug와 동일
        self.aug_map = {}
        img_paths = glob.glob(osp.join(self.train_dir, 'bounding_box_train', '*.jpg'))
        for img_path in img_paths:
            base = os.path.basename(img_path)
            aug_paths = []
            for i in range(1, 6):
                aug_path = osp.join(self.train_dir, 'aug_duke_inpainting_5prompts', base.replace('.jpg', f'_prompt_{i}.jpg'))
                if os.path.exists(aug_path):
                    aug_paths.append(aug_path)
            if aug_paths:
                self.aug_map[img_path] = aug_paths

    def clean_img_path(self, img_path):
        """
        `img_path`에서 in_distribution / out_distribution 경로와 _prompt_* 부분을 제거.
        
        Args:
            img_path (str): 원본 이미지 경로
        
        Returns:
            str: 정리된 이미지 경로
        """
        # Extract the original filename part from the augmented path
        base_filename = os.path.basename(img_path)
        # Remove suffixes like _prompt_N or _augmented_N
        original_filename_part = re.sub(r'(_prompt_|_augmented_)\d+', '', base_filename)

        # Construct the path to the original image in bounding_box_train
        original_dir = os.path.join(self.dataset_dir, 'bounding_box_train')

        # 새로운 경로 반환
        return os.path.join(original_dir, original_filename_part)

    def process_dir(self, dir_path, relabel=False):
        aug_dir_path = None # Initialize aug_dir_path
        if 'query' in dir_path:
            img_paths = glob.glob(osp.join(dir_path, '*.jpg'))  # query & gallery는 바로 찾기
            all_img_paths = img_paths
        elif 'bounding_box_test' in dir_path:
            img_paths = glob.glob(osp.join(dir_path, '*.jpg'))  # query & gallery는 바로 찾기
            all_img_paths = img_paths
            if len(all_img_paths) == 0:
                raise RuntimeError("No images found in gallery directory: " + dir_path)
        else:
            # Market1501_Aug와 동일한 방식으로 구현 
            img_paths = glob.glob(osp.join(dir_path, 'bounding_box_train', '*.jpg'))
            # Load augmented images from the specified directory
            aug_dir_path = osp.join(dir_path, 'aug_duke_inpainting_5prompts') # 증강 이미지 디렉토리 경로
            aug_paths = glob.glob(osp.join(aug_dir_path, '*.jpg')) # Find all jpgs directly inside

            all_img_paths = img_paths + aug_paths

        pid_container = set()
        for img_path in all_img_paths:
            pid, _ = map(int, self.pattern.search(img_path).groups())
            if pid == -1:
                continue # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in all_img_paths:
            pid, camid = map(int, self.pattern.search(img_path).groups())
            if pid == -1:
                continue # junk images are just ignored
            assert 1 <= camid <= 8  # OccludedDuke는 8개의 카메라 사용
            camid -= 1 # index starts from 0
            if relabel:
                pid = pid2label[pid]

            # Market1501_Aug와 동일한 방식으로 증강 이미지 처리
            # Check if the image path belongs to the augmented directory
            is_augmented = aug_dir_path is not None and osp.dirname(img_path) == aug_dir_path
            if is_augmented:
                clean_img_path = self.clean_img_path(img_path)
            else:
                clean_img_path = img_path # Use the original path directly

            masks_path = self.infer_masks_path(clean_img_path, self.masks_dir, self.masks_suffix)
            kp_path = self.infer_kp_path(clean_img_path)
            data.append({'img_path': img_path,
                         'pid': pid,
                         'masks_path': masks_path,
                         'camid': camid,
                         'kp_path': kp_path,
                         'is_augmented': is_augmented # Add flag to indicate if the image is augmented
                         })

        return data

    @staticmethod
    def filename_to_pid_camid(pattern, img_path):
        """
        파일 이름에서 pid와 camid 추출
        """
        pid, camid = map(int, pattern.search(img_path).groups())
        return pid, camid