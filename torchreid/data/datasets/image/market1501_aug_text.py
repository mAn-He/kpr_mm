from __future__ import division, print_function, absolute_import
# import ipdb
import re
import glob
import os.path as osp
import warnings
import os
from ..dataset import ImageDataset


class Market1501_Aug(ImageDataset):
    """Market1501.

    Reference:
        Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: `<http://www.liangzheng.org/Project/project_reid.html>`_
    
    Dataset statistics:
        - identities: 1501 (+1 for background).
        - images: 12936 (train) + 3368 (query) + 15913 (gallery).
    """
    _junk_pids = [0, -1]
    dataset_dir = 'Market-1501-v15.09.15'
    masks_base_dir = 'masks'
    dataset_url = 'http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip'
    cam_num = 6
    train_dir = 'bounding_box_train'
    # aug_dir = ''
    query_dir = 'query'
    gallery_dir = 'bounding_box_test'

    masks_dirs = {
        # dir_name: (parts_num, masks_stack_size, contains_background_mask)
        'pifpaf': (36, False, '.jpg.confidence_fields.npy'),
        'pifpaf_maskrcnn_filtering': (36, False, '.npy'),
    }

    @staticmethod
    def get_masks_config(masks_dir):
        if masks_dir not in Market1501_Aug.masks_dirs:
            return None
        else:
            return Market1501_Aug.masks_dirs[masks_dir]

    def __init__(self, root='', market1501_500k=False, masks_dir=None, **kwargs):
        self.kp_dir = kwargs['config'].model.kpr.keypoints.kp_dir
        self.masks_dir = masks_dir
        if self.masks_dir in self.masks_dirs:
            self.masks_parts_numbers, self.has_background, self.masks_suffix = self.masks_dirs[self.masks_dir]
        else:
            self.masks_parts_numbers, self.has_background, self.masks_suffix = None, None, None
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.download_dataset(self.dataset_dir, self.dataset_url)
        self.masks_dir = masks_dir

        # allow alternative directory structure
        if not osp.isdir(self.dataset_dir):
            warnings.warn(
                'The current data structure is deprecated. Please '
                'put data folders such as "bounding_box_train" under '
                '"Market-1501-v15.09.15".'
            )

        # self.train_dir = osp.join(self.dataset_dir, self.train_dir)
        # aug 넣으려고
        self.train_dir = self.dataset_dir
        self.query_dir = osp.join(self.dataset_dir, self.query_dir)
        self.gallery_dir = osp.join(self.dataset_dir, self.gallery_dir)
        self.extra_gallery_dir = osp.join(self.dataset_dir, 'images')
        self.market1501_500k = market1501_500k

        required_files = [
            self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]
        if self.market1501_500k:
            required_files.append(self.extra_gallery_dir)
        self.check_before_run(required_files)
        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)
        if self.market1501_500k:
            gallery += self.process_dir(self.extra_gallery_dir, relabel=False)
        super(Market1501_Aug, self).__init__(train, query, gallery, **kwargs)
        # 새로 추가한 부분
        self.aug_map = {}
        img_paths = glob.glob(osp.join(self.train_dir, '*.jpg'))
        for img_path in img_paths:
            base = os.path.basename(img_path)
            aug_paths = []
            for i in range(1, 6):
                aug_path = osp.join(self.train_dir, base.replace('.jpg', f'_prompt_{i}.jpg'))
                if os.path.exists(aug_path):
                    aug_paths.append(aug_path)
            if aug_paths:
                self.aug_map[img_path] = aug_paths

    def clean_img_path(self,img_path):
        """
        `img_path`에서 in_distribution / out_distribution 경로와 _prompt_* 부분을 제거.
        
        Args:
            img_path (str): 원본 이미지 경로
        
        Returns:
            str: 정리된 이미지 경로
        """
        # 1️⃣ `in_distribution` 또는 `out_distribution`이 포함된 경우 제거
        # Extract the original filename part from the augmented path
        # Assumes the augmented path structure is like /path/to/aug_market_inpainting_5prompts/original_filename_prompt_number.jpg
        # We need to get 'original_filename.jpg' and join it with the base train directory.
        base_filename = os.path.basename(img_path)
        # Remove suffixes like _prompt_N or _augmented_N
        original_filename_part = re.sub(r'(_prompt_|_augmented_)\d+', '', base_filename)

        # 2️⃣ `_prompt_*` 부분 제거 (파일명에서 `_prompt_숫자` 부분 삭제) - This comment is now less accurate but kept for context
        # Construct the path to the original image in bounding_box_train
        original_dir = os.path.join(self.dataset_dir, 'bounding_box_train')

        # 3️⃣ 새로운 경로 반환
        return os.path.join(original_dir, original_filename_part)
    def process_dir(self, dir_path, relabel=False):
        aug_dir_path = None # Initialize aug_dir_path
        if 'query' in dir_path:
            img_paths = glob.glob(osp.join(dir_path, '*.jpg'))  # query & gallery는 바로 찾기
            all_img_paths = img_paths
        elif  'bounding_box_test' in dir_path:
            img_paths = glob.glob(osp.join(dir_path, '*.jpg'))  # query & gallery는 바로 찾기
            all_img_paths = img_paths
            if len(all_img_paths) == 0:
                raise RuntimeError("No images found in gallery directory: " + dir_path)
        else:
        # img_paths = glob.glob(osp.join(dir_path, 'bounding_box_train', '*.jpg'))
            img_paths = glob.glob(osp.join(dir_path,'bounding_box_train', '*.jpg'))
            # Load augmented images from the specified directory
            aug_dir_path = osp.join(dir_path, 'aug_market_inpainting_5prompts') # Use the new directory name
            aug_paths = glob.glob(osp.join(aug_dir_path, '*.jpg')) # Find all jpgs directly inside

            all_img_paths = img_paths + aug_paths
        pattern = re.compile(r'([-\d]+)_c(\d)')
        # pattern = re.compile(r'(\d+)_c(\d+)s\d+_\d+.*\\.jpg')
        # pattern = re.compile(r'^(\d+)_c(\d+)(?=s).*\\.jpg')
        # ipdb.set_trace()
        # print(all_img_paths)
        pid_container = set()
        for img_path in all_img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        #마스크는 그대로
        for img_path in all_img_paths:
            # if 'prompt' in img_path:
            # img_path = self.clean_img_path(img_path)
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue # junk images are just ignored
            assert 0 <= pid <= 1501 # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1 # index starts from 0
            if relabel:
                pid = pid2label[pid]

            # Determine the base path for masks/keypoints: original path if not augmented, cleaned path if augmented
            # Check if the image path belongs to the augmented directory
            # Check if aug_dir_path is defined and the image path starts with it
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
        # Note: The 9:1 batch ratio for training (9 original, 1 augmented)
        # needs to be implemented in the DataLoader's sampler or batch_sampler
        # that uses this dataset. This class now provides the 'is_augmented'
        # flag for each sample to facilitate this.
        return data

    # def process_dir(self, dir_path, relabel=False):
    #     aug_dir_path = None # Initialize aug_dir_path
    #     img_paths = glob.glob(osp.join(dir_path,'bounding_box_train', '*.jpg'))
    #     # Load augmented images from the specified directory
    #     aug_dir_path = osp.join(dir_path, 'aug_market_inpainting_5prompts') # Use the new directory name
    #     aug_paths = glob.glob(osp.join(aug_dir_path, '*.jpg')) # Find all jpgs directly inside

    #     all_img_paths = img_paths + aug_paths
    #     pattern = re.compile(r'([-\d]+)_c(\d)')
    #     # pattern = re.compile(r'(\d+)_c(\d+)s\d+_\d+.*\\.jpg')
    #     # pattern = re.compile(r'^(\d+)_c(\d+)(?=s).*\\.jpg')
    #     # ipdb.set_trace()
    #     # print(all_img_paths)
    #     pid_container = set()
    #     for img_path in all_img_paths:
    #         pid, _ = map(int, pattern.search(img_path).groups())
    #         if pid == -1:
    #             continue # junk images are just ignored
    #         pid_container.add(pid)
    #     pid2label = {pid: label for label, pid in enumerate(pid_container)}

    #     data = []
    #     #마스크는 그대로
    #     for img_path in all_img_paths:
    #         # if 'prompt' in img_path:
    #         # img_path = self.clean_img_path(img_path)
    #         pid, camid = map(int, pattern.search(img_path).groups())
    #         if pid == -1:
    #             continue # junk images are just ignored
    #         assert 0 <= pid <= 1501 # pid == 0 means background
    #         assert 1 <= camid <= 6
    #         camid -= 1 # index starts from 0
    #         if relabel:
    #             pid = pid2label[pid]
    #
    #         # Determine the base path for masks/keypoints: original path if not augmented, cleaned path if augmented
    #         # Check if the image path belongs to the augmented directory
    #         is_augmented = osp.dirname(img_path) == aug_dir_path
    #         if is_augmented:
    #             clean_img_path = self.clean_img_path(img_path)
    #         else:
    #             clean_img_path = img_path # Use the original path directly
    #
    #         masks_path = self.infer_masks_path(clean_img_path, self.masks_dir, self.masks_suffix)
    #         kp_path = self.infer_kp_path(clean_img_path)
    #         data.append({'img_path': img_path,
    #                      'pid': pid,
    #                      'masks_path': masks_path,
    #                      'camid': camid,
    #                      'kp_path': kp_path,
    #                      })
    #     return data
