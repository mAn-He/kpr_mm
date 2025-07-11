from __future__ import division, print_function, absolute_import
import torch
from torch.utils.data import BatchSampler
# from torchreid.data.sampler_aug import AugmentedRatioBatchSampler
from torchreid.data.masks_transforms import masks_preprocess_transforms
from torchreid.data.sampler import build_train_sampler,wrap_with_batch_pids,AugmentedRatioBatchSampler
from torchreid.data.datasets import init_image_dataset, init_video_dataset, get_image_dataset
from torchreid.data.transforms import build_transforms
import os.path as osp
from functools import partial
from torchreid.data.collate_aug import custom_collate_fn

class DataManager(object):
    r"""Base data manager.

    Args:
        sources (str or list): source dataset(s).
        targets (str or list, optional): target dataset(s). If not given,
            it equals to ``sources``.
        height (int, optional): target image height. Default is 256.
        width (int, optional): target image width. Default is 128.
        transforms (str or list of str, optional): transformations applied to model training.
            Default is 'random_flip'.
        norm_mean (list or None, optional): data mean. Default is None (use imagenet mean).
        norm_std (list or None, optional): data std. Default is None (use imagenet std).
        use_gpu (bool, optional): use gpu. Default is True.
    """

    def __init__(
        self,
        config,
        sources=None,
        targets=None,
        height=256,
        width=128,
        transforms='random_flip',
        norm_mean=None,
        norm_std=None,
        use_gpu=False,
        masks_dir='',
        root='',
    ):
        self.sources = sources
        self.targets = targets
        self.height = height
        self.width = width
        self.masks_dir = masks_dir
        self.config = config

        if self.sources is None:
            raise ValueError('sources must not be None')

        if isinstance(self.sources, str):
            self.sources = [self.sources]

        if self.targets is None:
            self.targets = self.sources

        if isinstance(self.targets, str):
            self.targets = [self.targets]

        train_dataset = get_image_dataset(self.sources[0])
        masks_config = train_dataset.get_masks_config(self.masks_dir)
        self.transform_tr, self.transform_te, self.kp_target_transform, self.kp_prompt_transform = build_transforms(
            self.height,
            self.width,
            config,
            transforms=transforms,
            norm_mean=norm_mean,
            norm_std=norm_std,
            remove_background_mask=masks_config[1] if masks_config is not None else False,
            masks_preprocess=config.model.kpr.masks.preprocess,
            softmax_weight=config.model.kpr.masks.softmax_weight,
            background_computation_strategy=config.model.kpr.masks.background_computation_strategy,
            mask_filtering_threshold=config.model.kpr.masks.mask_filtering_threshold,
            train_dir=osp.join(osp.join(osp.abspath(osp.expanduser(root)), train_dataset.dataset_dir), train_dataset.train_dir),
        )

        self.use_gpu = (torch.cuda.is_available() and use_gpu)

    @property
    def num_train_pids(self):
        """Returns the number of training person identities."""
        return self._num_train_pids

    @property
    def num_train_cams(self):
        """Returns the number of training cameras."""
        return self._num_train_cams

    def fetch_test_loaders(self, name):
        """Returns query and gallery of a test dataset, each containing
        tuples of (img_path(s), pid, camid).

        Args:
            name (str): dataset name.
        """
        query_loader = self.test_dataset[name]['query']
        gallery_loader = self.test_dataset[name]['gallery']
        return query_loader, gallery_loader

    def preprocess_pil_img(self, img):
        """Transforms a PIL image to torch tensor for testing."""
        return self.transform_te(img)


class ImageDataManager(DataManager):
    r"""Image data manager.

    Args:
        root (str): root path to datasets.
        sources (str or list): source dataset(s).
        targets (str or list, optional): target dataset(s). If not given,
            it equals to ``sources``.
        height (int, optional): target image height. Default is 256.
        width (int, optional): target image width. Default is 128.
        transforms (str or list of str, optional): transformations applied to model training.
            Default is 'random_flip'.
        norm_mean (list or None, optional): data mean. Default is None (use imagenet mean).
        norm_std (list or None, optional): data std. Default is None (use imagenet std).
        use_gpu (bool, optional): use gpu. Default is True.
        split_id (int, optional): split id (*0-based*). Default is 0.
        combineall (bool, optional): combine train, query and gallery in a dataset for
            training. Default is False.
        load_train_targets (bool, optional): construct train-loader for target datasets.
            Default is False. This is useful for domain adaptation research.
        batch_size_train (int, optional): number of images in a training batch. Default is 32.
        batch_size_test (int, optional): number of images in a test batch. Default is 32.
        workers (int, optional): number of workers. Default is 4.
        num_instances (int, optional): number of instances per identity in a batch.
            Default is 4.
        train_sampler (str, optional): sampler. Default is RandomSampler.
        train_sampler_t (str, optional): sampler for target train loader. Default is RandomSampler.
        cuhk03_labeled (bool, optional): use cuhk03 labeled images.
            Default is False (defaul is to use detected images).
        cuhk03_classic_split (bool, optional): use the classic split in cuhk03.
            Default is False.
        market1501_500k (bool, optional): add 500K distractors to the gallery
            set in market1501. Default is False.

    Examples::

        datamanager = torchreid.data.ImageDataManager(
            root='path/to/reid-data',
            sources='market1501',
            height=256,
            width=128,
            batch_size_train=32,
            batch_size_test=100
        )

        # return train loader of source data
        train_loader = datamanager.train_loader

        # return test loader of target data
        test_loader = datamanager.test_loader

        # return train loader of target data
        train_loader_t = datamanager.train_loader_t
    """
    data_type = 'image'

    def __init__(
        self,
        config,
        root='',
        sources=None,
        targets=None,
        height=256,
        width=128,
        transforms='random_flip',
        norm_mean=None,
        norm_std=None,
        use_gpu=True,
        split_id=0,
        combineall=False,
        load_train_targets=False,
        batch_size_train=32,
        batch_size_test=32,
        workers=4,
        num_instances=4,
        train_sampler='RandomSampler',
        train_sampler_t='RandomSampler',
        cuhk03_labeled=False,
        cuhk03_classic_split=False,
        market1501_500k=False,
        masks_dir=None,
        aug_ratio=0.0,
    ):

        super(ImageDataManager, self).__init__(
            config=config,
            sources=sources,
            targets=targets,
            height=height,
            width=width,
            transforms=transforms,
            norm_mean=norm_mean,
            norm_std=norm_std,
            use_gpu=use_gpu,
            masks_dir=masks_dir,
            root=root,
        )
        random_occlusions = 'bipo' in transforms or 'bipo_test' in transforms or 'bipot' in transforms
        print('=> Loading train (source) dataset')
        trainset = []
        for name in self.sources:
            trainset_ = init_image_dataset(
                name,
                config=config,
                transform_tr=self.transform_tr,
                transform_te=self.transform_te,
                kp_target_transform=self.kp_target_transform,
                kp_prompt_transform=self.kp_prompt_transform,
                mode='train',
                combineall=combineall,
                root=root,
                split_id=split_id,
                cuhk03_labeled=cuhk03_labeled,
                cuhk03_classic_split=cuhk03_classic_split,
                market1501_500k=market1501_500k,
                masks_dir=masks_dir,
                load_masks=self.config.model.kpr.masks.preprocess in masks_preprocess_transforms or self.config.model.kpr.masks.preprocess == 'none',  # none to load masks and apply no grouping
                random_occlusions=random_occlusions,
            )
            trainset.append(trainset_)
        trainset = sum(trainset)

        self._num_train_pids = trainset.num_train_pids
        self._num_train_cams = trainset.num_train_cams
        id_sampler = build_train_sampler(
            trainset.train, 'RandomIdentitySampler',
            batch_size=batch_size_train,
            num_instances=num_instances
        )

        # ② BatchSampler: 인덱스를 batch_size 단위로 묶음
        id_batch_sampler = BatchSampler(id_sampler,
                                        batch_size=batch_size_train,
                                        drop_last=True)

        # ③ AugmentedRatioBatchSampler: 원본 batch + 증강
        if self.config.data.aug_ratio > 0:
            base_batch_sampler = AugmentedRatioBatchSampler(
                trainset.train, id_batch_sampler,
                aug_ratio=self.config.data.aug_ratio
            )
        else:
            base_batch_sampler = id_batch_sampler

        # ④ BatchPidWrapper: BIPO용 (idx, batch_pids) 튜플로 변환
        need_pids = self.config.data.bipo.get("pid_sampling_from_batch", False)
        batch_sampler = wrap_with_batch_pids(base_batch_sampler,
                                            trainset, enable=need_pids)

        # DataLoader
        self.train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_sampler=batch_sampler,
            num_workers=workers,
            pin_memory=self.use_gpu,
            collate_fn=partial(custom_collate_fn, aug_map=getattr(trainset, 'aug_map', {}), aug_ratio=self.config.data.aug_ratio)
        )

        self.train_loader_t = None
        if load_train_targets:
            # check if sources and targets are identical
            assert len(set(self.sources) & set(self.targets)) == 0, \
                'sources={} and targets={} must not have overlap'.format(self.sources, self.targets)

            print('=> Loading train (target) dataset')
            trainset_t = []
            for name in self.targets:
                trainset_t_ = init_image_dataset(
                    name,
                    config=config,
                    transform_tr=self.transform_tr,
                    transform_te=self.transform_te,
                    kp_target_transform=self.kp_target_transform,
                    kp_prompt_transform=self.kp_prompt_transform,
                    mode='train',
                    combineall=False, # only use the training data
                    root=root,
                    split_id=split_id,
                    cuhk03_labeled=cuhk03_labeled,
                    cuhk03_classic_split=cuhk03_classic_split,
                    market1501_500k=market1501_500k,
                    masks_dir=masks_dir,
                    load_masks=self.config.model.kpr.masks.preprocess in masks_preprocess_transforms,
                )
                trainset_t.append(trainset_t_)
            trainset_t = sum(trainset_t)

            self.train_loader_t = torch.utils.data.DataLoader(
                trainset_t,
                sampler=build_train_sampler(
                    trainset_t.train,
                    train_sampler_t,
                    batch_size=batch_size_train,
                    num_instances=num_instances
                ),
                batch_size=batch_size_train,
                shuffle=False,
                num_workers=workers,
                pin_memory=self.use_gpu,
                drop_last=True
            )

        print('=> Loading test (target) dataset')
        self.test_loader = {
            name: {
                'query': None,
                'gallery': None
            }
            for name in self.targets
        }
        self.test_dataset = {
            name: {
                'query': None,
                'gallery': None
            }
            for name in self.targets
        }

        for name in self.targets:
            # build query loader
            queryset = init_image_dataset(
                name,
                config=config,
                transform_tr=self.transform_tr,
                transform_te=self.transform_te,
                kp_target_transform=self.kp_target_transform,
                kp_prompt_transform=self.kp_prompt_transform,
                mode='query',
                combineall=combineall,
                root=root,
                split_id=split_id,
                cuhk03_labeled=cuhk03_labeled,
                cuhk03_classic_split=cuhk03_classic_split,
                market1501_500k=market1501_500k,
                masks_dir=masks_dir,
                load_masks=self.config.model.kpr.masks.preprocess in masks_preprocess_transforms,
            )
            self.test_loader[name]['query'] = torch.utils.data.DataLoader(
                queryset,
                batch_size=batch_size_test,
                shuffle=False,
                num_workers=workers,
                pin_memory=self.use_gpu,
                drop_last=False
            )

            # build gallery loader
            galleryset = init_image_dataset(
                name,
                config=config,
                transform_tr=self.transform_tr,
                transform_te=self.transform_te,
                kp_target_transform=self.kp_target_transform,
                kp_prompt_transform=self.kp_prompt_transform,
                mode='gallery',
                combineall=combineall,
                verbose=False,
                root=root,
                split_id=split_id,
                cuhk03_labeled=cuhk03_labeled,
                cuhk03_classic_split=cuhk03_classic_split,
                market1501_500k=market1501_500k,
                masks_dir=masks_dir,
                load_masks=self.config.model.kpr.masks.preprocess in masks_preprocess_transforms,
            )
            self.test_loader[name]['gallery'] = torch.utils.data.DataLoader(
                galleryset,
                batch_size=batch_size_test,
                shuffle=False,
                num_workers=workers,
                pin_memory=self.use_gpu,
                drop_last=False
            )

            self.test_dataset[name]['query'] = queryset.query
            self.test_dataset[name]['gallery'] = galleryset.gallery

        print('\n')
        print('  **************** Summary ****************')
        print('  source            : {}'.format(self.sources))
        print('  # source datasets : {}'.format(len(self.sources)))
        print('  # source ids      : {}'.format(self.num_train_pids))
        print('  # source images   : {}'.format(len(trainset)))
        print('  # source cameras  : {}'.format(self.num_train_cams))
        if load_train_targets:
            print(
                '  # target images   : {} (unlabeled)'.format(len(trainset_t))
            )
        print('  target            : {}'.format(self.targets))
        print('  *****************************************')
        print('\n')


class VideoDataManager(DataManager):
    r"""Video data manager.

    Args:
        root (str): root path to datasets.
        sources (str or list): source dataset(s).
        targets (str or list, optional): target dataset(s). If not given,
            it equals to ``sources``.
        height (int, optional): target image height. Default is 256.
        width (int, optional): target image width. Default is 128.
        transforms (str or list of str, optional): transformations applied to model training.
            Default is 'random_flip'.
        norm_mean (list or None, optional): data mean. Default is None (use imagenet mean).
        norm_std (list or None, optional): data std. Default is None (use imagenet std).
        use_gpu (bool, optional): use gpu. Default is True.
        split_id (int, optional): split id (*0-based*). Default is 0.
        combineall (bool, optional): combine train, query and gallery in a dataset for
            training. Default is False.
        batch_size_train (int, optional): number of tracklets in a training batch. Default is 3.
        batch_size_test (int, optional): number of tracklets in a test batch. Default is 3.
        workers (int, optional): number of workers. Default is 4.
        num_instances (int, optional): number of instances per identity in a batch.
            Default is 4.
        train_sampler (str, optional): sampler. Default is RandomSampler.
        seq_len (int, optional): how many images to sample in a tracklet. Default is 15.
        sample_method (str, optional): how to sample images in a tracklet. Default is "evenly".
            Choices are ["evenly", "random", "all"]. "evenly" and "random" will sample ``seq_len``
            images in a tracklet while "all" samples all images in a tracklet, where the batch size
            needs to be set to 1.

    Examples::

        datamanager = torchreid.data.VideoDataManager(
            root='path/to/reid-data',
            sources='mars',
            height=256,
            width=128,
            batch_size_train=3,
            batch_size_test=3,
            seq_len=15,
            sample_method='evenly'
        )

        # return train loader of source data
        train_loader = datamanager.train_loader

        # return test loader of target data
        test_loader = datamanager.test_loader

    .. note::
        The current implementation only supports image-like training. Therefore, each image in a
        sampled tracklet will undergo independent transformation functions. To achieve tracklet-aware
        training, you need to modify the transformation functions for video reid such that each function
        applies the same operation to all images in a tracklet to keep consistency.
    """
    data_type = 'video'

    def __init__(
        self,
        config,
        root='',
        sources=None,
        targets=None,
        height=256,
        width=128,
        transforms='random_flip',
        norm_mean=None,
        norm_std=None,
        use_gpu=True,
        split_id=0,
        combineall=False,
        batch_size_train=3,
        batch_size_test=3,
        workers=4,
        num_instances=4,
        train_sampler='RandomSampler',
        seq_len=15,
        sample_method='evenly'
    ):

        super(VideoDataManager, self).__init__(
            config=config,
            sources=sources,
            targets=targets,
            height=height,
            width=width,
            transforms=transforms,
            norm_mean=norm_mean,
            norm_std=norm_std,
            use_gpu=use_gpu
        )

        print('=> Loading train (source) dataset')
        trainset = []
        for name in self.sources:
            trainset_ = init_video_dataset(
                name,
                transform=self.transform_tr,
                mode='train',
                combineall=combineall,
                root=root,
                split_id=split_id,
                seq_len=seq_len,
                sample_method=sample_method
            )
            trainset.append(trainset_)
        trainset = sum(trainset)

        self._num_train_pids = trainset.num_train_pids
        self._num_train_cams = trainset.num_train_cams

        train_sampler = build_train_sampler(
            trainset.train,
            train_sampler,
            batch_size=batch_size_train,
            num_instances=num_instances
        )

        self.train_loader = torch.utils.data.DataLoader(
            trainset,
            sampler=train_sampler,
            batch_size=batch_size_train,
            shuffle=False,
            num_workers=workers,
            pin_memory=self.use_gpu,
            drop_last=True
        )

        print('=> Loading test (target) dataset')
        self.test_loader = {
            name: {
                'query': None,
                'gallery': None
            }
            for name in self.targets
        }
        self.test_dataset = {
            name: {
                'query': None,
                'gallery': None
            }
            for name in self.targets
        }

        for name in self.targets:
            # build query loader
            queryset = init_video_dataset(
                name,
                transform=self.transform_te,
                mode='query',
                combineall=combineall,
                root=root,
                split_id=split_id,
                seq_len=seq_len,
                sample_method=sample_method
            )
            self.test_loader[name]['query'] = torch.utils.data.DataLoader(
                queryset,
                batch_size=batch_size_test,
                shuffle=False,
                num_workers=workers,
                pin_memory=self.use_gpu,
                drop_last=False
            )

            # build gallery loader
            galleryset = init_video_dataset(
                name,
                transform=self.transform_te,
                mode='gallery',
                combineall=combineall,
                verbose=False,
                root=root,
                split_id=split_id,
                seq_len=seq_len,
                sample_method=sample_method
            )
            self.test_loader[name]['gallery'] = torch.utils.data.DataLoader(
                galleryset,
                batch_size=batch_size_test,
                shuffle=False,
                num_workers=workers,
                pin_memory=self.use_gpu,
                drop_last=False
            )

            self.test_dataset[name]['query'] = queryset.query
            self.test_dataset[name]['gallery'] = galleryset.gallery

        print('\n')
        print('  **************** Summary ****************')
        print('  source             : {}'.format(self.sources))
        print('  # source datasets  : {}'.format(len(self.sources)))
        print('  # source ids       : {}'.format(self.num_train_pids))
        print('  # source tracklets : {}'.format(len(trainset)))
        print('  # source cameras   : {}'.format(self.num_train_cams))
        print('  target             : {}'.format(self.targets))
        print('  *****************************************')
        print('\n')
