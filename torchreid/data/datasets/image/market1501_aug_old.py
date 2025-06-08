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
    def clean_img_path(self,img_path):
        """
        `img_path`에서 in_distribution / out_distribution 경로와 _prompt_* 부분을 제거.
        
        Args:
            img_path (str): 원본 이미지 경로
        
        Returns:
            str: 정리된 이미지 경로
        """
        # 1️⃣ `in_distribution` 또는 `out_distribution`이 포함된 경우 제거
        # img_path = re.sub(r'[/\\](in_distribution|out_distribution)[/\\]', '/', img_path)
        img_path = re.sub(r'[/\\]home[/\\]hseung[/\\]keypoint_promptable_reidentification[/\\]Market-1501-v15.09.15[/\\]aug_market_inpainting_turbo_realistic[/\\](in_distribution|out_distribution)[/\\]', '/', img_path)


        # 2️⃣ `_prompt_*` 부분 제거 (파일명에서 `_prompt_숫자` 부분 삭제)
        dir_name, filename = os.path.split(img_path)
        filename = re.sub(r'_prompt_\d+', '', filename)  # `_prompt_숫자` 제거

        # 3️⃣ 새로운 경로 반환
        return os.path.join(dir_name,'bounding_box_train',filename)
    def process_dir(self, dir_path, relabel=False):
        # query나 gallery 디렉토리 처리
        if 'query' in dir_path or 'bounding_box_test' in dir_path:
            img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
            all_img_paths = img_paths
        # train 디렉토리 처리
        else:
            img_paths = glob.glob(osp.join(dir_path, 'bounding_box_train', '*.jpg'))
            aug_in = glob.glob(osp.join(dir_path, 'aug_market_inpainting_turbo_realistic', 'in_distribution', '*.jpg'))
            aug_out = glob.glob(osp.join(dir_path, 'aug_market_inpainting_turbo_realistic', 'out_of_distribution', '*.jpg'))
            all_img_paths = img_paths + aug_in + aug_out

        pattern = re.compile(r'([-\d]+)_c(\d)')

        # PID 컨테이너 생성
        pid_container = set()
        for img_path in all_img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in all_img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]

            # 마스크와 키포인트 경로 설정
            if 'prompt' in img_path:
                # augmented 이미지의 경우 원본 이미지 경로로 변환
                clean_img_path = self.clean_img_path(img_path)
                masks_path = self.infer_masks_path(clean_img_path, self.masks_dir, self.masks_suffix)
                kp_path = self.infer_kp_path(clean_img_path)
            else:
                # 원본 이미지는 그대로 사용
                masks_path = self.infer_masks_path(img_path, self.masks_dir, self.masks_suffix)
                kp_path = self.infer_kp_path(img_path)

            data.append({
                'img_path': img_path,
                'pid': pid,
                'masks_path': masks_path,
                'camid': camid,
                'kp_path': kp_path,
            })

        return data

    # def process_dir(self, dir_path, relabel=False):
    #     img_paths = glob.glob(osp.join(dir_path,'bounding_box_train', '*.jpg'))
    #     aug_in = glob.glob(osp.join(dir_path, 'aug_market_inpainting_turbo_realistic', 'in_distribution', '*.jpg'))
    #     aug_out = glob.glob(osp.join(dir_path, 'aug_market_inpainting_turbo_realistic', 'out_of_distribution', '*.jpg'))

    #     all_img_paths = img_paths + aug_in + aug_out
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
    #         if 'prompt' in img_path:
    #             clean_img_path = self.clean_img_path(img_path)
    #         masks_path = self.infer_masks_path(clean_img_path, self.masks_dir, self.masks_suffix)
    #         kp_path = self.infer_kp_path(clean_img_path)
    #         data.append({'img_path': img_path,
    #                      'pid': pid,
    #                      'masks_path': masks_path,
    #                      'camid': camid,
    #                      'kp_path': kp_path,
    #                      })
    #     return data
=========