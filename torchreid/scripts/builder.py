import os
import torch
import torch.nn as nn
import torchreid
import pathlib
from torchreid.data import compute_parts_num_and_names
from torchreid.models.promptable_timm_swin import swin_timm_models
from torch.nn.parallel import DistributedDataParallel as DDP

from torchreid.utils import (
    Logger,
    check_isfile,
    set_random_seed,
    resume_from_checkpoint,
    load_pretrained_weights,
    compute_model_complexity,
    Writer,
    load_checkpoint,
)

from torchreid.scripts.default_config import (
    imagedata_kwargs,
    optimizer_kwargs,
    videodata_kwargs,
    get_default_config,
    lr_scheduler_kwargs,
    display_config_diff,
)
from torchreid.utils.engine_state import EngineState
# 청킹 엔진 import 추가
from torchreid.engine.image.part_based_engine_chunked import ImagePartBasedEngineChunked
# TODO: SuperCon 청킹 엔진 구현 후 주석 해제
# from torchreid.engine.image.supcontrastive_engine_chunked import ImageSupConEngineChunked


def build_datamanager(cfg, test_only=False):
    # print(f"🔍 [BUILDER] TEXT DEBUG - Building datamanager...")
    # print(f"🔍 [BUILDER] TEXT DEBUG - cfg.data.type: {cfg.data.type}")
    # print(f"🔍 [BUILDER] TEXT DEBUG - hasattr(cfg.model, 'text_prompt'): {hasattr(cfg.model, 'text_prompt')}")

    # Check for text configuration
    text_enabled = False
    text_config = None

    if hasattr(cfg.model, 'text_prompt'):
        text_enabled = getattr(cfg.model.text_prompt, 'enabled', False)
        text_config = cfg.model.text_prompt if text_enabled else None
        # print(f"🔍 [BUILDER] TEXT DEBUG - text_enabled: {text_enabled}")
        # print(f"🔍 [BUILDER] TEXT DEBUG - text_config: {text_config}")
    else:
        # print(f"🔍 [BUILDER] TEXT DEBUG - No text_prompt configuration found")
        pass

    # For zeroshot/test_only mode, don't load problematic source datasets
    if test_only and cfg.data.sources != cfg.data.targets:
        print(f"🔧 Test-only mode: Using targets {cfg.data.targets} as sources to avoid loading training data")
        original_sources = cfg.data.sources
        cfg.data.sources = cfg.data.targets  # Use targets as sources for test-only
        print(f"   Original sources: {original_sources} -> Modified sources: {cfg.data.sources}")

    if cfg.data.type == "image":
        kwargs = imagedata_kwargs(cfg)
        # Add text support parameters
        kwargs['text_enabled'] = text_enabled
        kwargs['text_config'] = text_config
        # print(f"🔍 [BUILDER] TEXT DEBUG - ImageDataManager kwargs updated with text support")
        return torchreid.data.ImageDataManager(**kwargs)
    else:
        kwargs = videodata_kwargs(cfg)
        # Add text support parameters for video (if needed in future)
        kwargs['text_enabled'] = text_enabled
        kwargs['text_config'] = text_config
        # print(f"🔍 [BUILDER] TEXT DEBUG - VideoDataManager kwargs updated with text support")
        return torchreid.data.VideoDataManager(**kwargs)


def build_engine(cfg, datamanager, model, optimizer, scheduler, writer, engine_state):
    if cfg.data.type == "image":
        if cfg.loss.name == "softmax":
            if cfg.train.accumulation_steps > 1:
                engine = torchreid.engine.ImageSoftmaxEngineAccumulation(
                    datamanager,
                    model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    use_gpu=cfg.use_gpu,
                    label_smooth=cfg.loss.softmax.label_smooth,
                    save_model_flag=cfg.model.save_model_flag,
                    writer=writer,
                    engine_state=engine_state,
                    accumulation_steps=cfg.train.accumulation_steps,
                )
            else:
                engine = torchreid.engine.ImageSoftmaxEngine(
                    datamanager,
                    model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    use_gpu=cfg.use_gpu,
                    label_smooth=cfg.loss.softmax.label_smooth,
                    save_model_flag=cfg.model.save_model_flag,
                    writer=writer,
                    engine_state=engine_state,
                )

        elif cfg.loss.name == "triplet":
            engine = torchreid.engine.ImageTripletEngine(
                datamanager,
                model,
                optimizer=optimizer,
                margin=cfg.loss.triplet.margin,
                weight_t=cfg.loss.triplet.weight_t,
                weight_x=cfg.loss.triplet.weight_x,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth,
                save_model_flag=cfg.model.save_model_flag,
                writer=writer,
                engine_state=engine_state,
            )
        elif cfg.loss.name == "part_based":
            # 🔄 청킹 엔진 vs 일반 엔진 선택 로직
            use_chunked_engine = cfg.train.accumulation_steps >= 2

            if use_chunked_engine:
                print(f"🚀 Using ChunkedEngine (accumulation_steps={cfg.train.accumulation_steps} >= 2)")
                chunk_size = getattr(cfg.train, 'chunk_size', cfg.train.batch_size // 2)

                if cfg.loss.supercon.enabled:
                    # SuperCon + 청킹 엔진 사용 (향후 구현)
                    print("⚠️ SuperCon + Chunking: using basic chunked engine with SuperCon loss disabled temporarily")
                    # TODO: ImageSupConEngineChunked 구현 후 사용
                    engine = ImagePartBasedEngineChunked(
                        datamanager,
                        model,
                        optimizer=optimizer,
                        loss_name=cfg.loss.part_based.name,
                        config=cfg,
                        margin=cfg.loss.triplet.margin,
                        scheduler=scheduler,
                        use_gpu=cfg.use_gpu,
                        save_model_flag=cfg.model.save_model_flag,
                        writer=writer,
                        engine_state=engine_state,
                        dist_combine_strat=cfg.test.part_based.dist_combine_strat,
                        batch_size_pairwise_dist_matrix=cfg.test.batch_size_pairwise_dist_matrix,
                        mask_filtering_training=cfg.model.kpr.mask_filtering_training,
                        mask_filtering_testing=cfg.model.kpr.mask_filtering_testing,
                        accumulation_steps=cfg.train.accumulation_steps,
                        chunk_size=chunk_size,
                    )
                else:
                    # 청킹 엔진 사용
                    engine = ImagePartBasedEngineChunked(
                        datamanager,
                        model,
                        optimizer=optimizer,
                        loss_name=cfg.loss.part_based.name,
                        config=cfg,
                        margin=cfg.loss.triplet.margin,
                        scheduler=scheduler,
                        use_gpu=cfg.use_gpu,
                        save_model_flag=cfg.model.save_model_flag,
                        writer=writer,
                        engine_state=engine_state,
                        dist_combine_strat=cfg.test.part_based.dist_combine_strat,
                        batch_size_pairwise_dist_matrix=cfg.test.batch_size_pairwise_dist_matrix,
                        mask_filtering_training=cfg.model.kpr.mask_filtering_training,
                        mask_filtering_testing=cfg.model.kpr.mask_filtering_testing,
                        accumulation_steps=cfg.train.accumulation_steps,
                        chunk_size=chunk_size,
                    )
            else:
                # 기존 엔진 사용
                if cfg.loss.supercon.enabled:
                    print(f"🔧 Using SuperCon engine (accumulation_steps={cfg.train.accumulation_steps} < 2)")
                    engine = torchreid.engine.ImageSupConEngine(
                        datamanager,
                        model,
                        optimizer=optimizer,
                        loss_name=cfg.loss.part_based.name,
                        config=cfg,
                        margin=cfg.loss.triplet.margin,
                        scheduler=scheduler,
                        use_gpu=cfg.use_gpu,
                        save_model_flag=cfg.model.save_model_flag,
                        writer=writer,
                        engine_state=engine_state,
                        dist_combine_strat=cfg.test.part_based.dist_combine_strat,
                        batch_size_pairwise_dist_matrix=cfg.test.batch_size_pairwise_dist_matrix,
                        mask_filtering_training=cfg.model.kpr.mask_filtering_training,
                        mask_filtering_testing=cfg.model.kpr.mask_filtering_testing,
                        accumulation_steps=cfg.train.accumulation_steps,
                    )
                else:
                    print(f"🔧 Using regular PartBased engine (accumulation_steps={cfg.train.accumulation_steps} < 2)")
                    engine = torchreid.engine.ImagePartBasedEngine(
                        datamanager,
                        model,
                        optimizer=optimizer,
                        loss_name=cfg.loss.part_based.name,
                        config=cfg,
                        margin=cfg.loss.triplet.margin,
                        scheduler=scheduler,
                        use_gpu=cfg.use_gpu,
                        save_model_flag=cfg.model.save_model_flag,
                        writer=writer,
                        engine_state=engine_state,
                        dist_combine_strat=cfg.test.part_based.dist_combine_strat,
                        batch_size_pairwise_dist_matrix=cfg.test.batch_size_pairwise_dist_matrix,
                        mask_filtering_training=cfg.model.kpr.mask_filtering_training,
                        mask_filtering_testing=cfg.model.kpr.mask_filtering_testing,
                        accumulation_steps=cfg.train.accumulation_steps,
                    )

    else:
        if cfg.loss.name == "softmax":
            engine = torchreid.engine.VideoSoftmaxEngine(
                datamanager,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth,
                pooling_method=cfg.video.pooling_method,
                save_model_flag=cfg.model.save_model_flag,
                writer=writer,
                engine_state=engine_state,
            )

        else:
            engine = torchreid.engine.VideoTripletEngine(
                datamanager,
                model,
                optimizer=optimizer,
                margin=cfg.loss.triplet.margin,
                weight_t=cfg.loss.triplet.weight_t,
                weight_x=cfg.loss.triplet.weight_x,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth,
                save_model_flag=cfg.model.save_model_flag,
                writer=writer,
                engine_state=engine_state,
            )

    return engine


def reset_config(cfg, args):
    if args.root:
        cfg.data.root = args.root
    if args.save_dir:
        cfg.data.save_dir = args.save_dir
    if args.inference_enabled:
        cfg.inference.enabled = args.inference_enabled
    if args.sources:
        cfg.data.sources = args.sources
    if args.targets:
        cfg.data.targets = args.targets
    if args.transforms:
        cfg.data.transforms = args.transforms
    if args.job_id:
        cfg.project.job_id = args.job_id
    if args.accumulation_steps:
        cfg.train.accumulation_steps = args.accumulation_steps

def build_config(args=None, config=None, config_path=None, display_diff=False):
    cfg = get_default_config()
    cfg.train.accumulation_steps = 1 # default accumulation steps
    default_cfg_copy = cfg.clone()

    # use gpu if available
    cfg.use_gpu = torch.cuda.is_available()

    # overwrite default configs with provided args
    if config_path is not None:
        cfg.merge_from_file(config_path)
        cfg.project.config_file = os.path.basename(config_path)
    if config is not None:
        cfg.merge_from_other_cfg(config)
    if args is not None:
        reset_config(cfg, args)
        cfg.merge_from_list(args.opts)

    compute_parts_num_and_names(cfg)

    if cfg.model.load_weights:
        assert check_isfile(
            cfg.model.load_weights
        ), "Non-existing weight file '{}'".format(cfg.model.load_weights)
    if (
        cfg.model.load_weights
        and check_isfile(cfg.model.load_weights)
        and cfg.model.load_config
    ):
        checkpoint = load_checkpoint(cfg.model.load_weights)
        if "config" in checkpoint:
            print(
                "Overwriting current config with config loaded from {}".format(
                    cfg.model.load_weights
                )
            )
            kpr_config = checkpoint["config"].model.kpr
            if checkpoint["config"].data.sources[0] != cfg.data.targets[0]:
                print(
                    "WARNING: the train dataset of the loaded model is different from the target dataset in the "
                    "current config."
                )
            kpr_config.pop("backbone_pretrained_path", None)
            kpr_config.masks.pop("dir", None)
            if "keypoints" in kpr_config:
                kpr_config.keypoints.pop("enabled", None)
                kpr_config.keypoints.pop("kp_dir", None)
            if cfg.model.discard_test_params:
                kpr_config.masks.pop("dir", None)
                kpr_config.masks.pop("preprocess", None)
                kpr_config.pop("mask_filtering_testing", None)
                kpr_config.pop("learnable_attention_enabled", None)
                kpr_config.pop("test_embeddings", None)
                kpr_config.pop("test_use_target_segmentation", None)
                kpr_config.pop("testing_binary_visibility_score", None)
            cfg.model.kpr.merge_from_other_cfg(kpr_config)
            if "vit" in checkpoint["config"].model:
                if cfg.model.discard_test_params:
                    checkpoint["config"].model.promptable_trans.pop("disable_inference_prompting", None)
                cfg.model.promptable_trans.merge_from_other_cfg(checkpoint["config"].model.promptable_trans)
            if "transreid" in checkpoint["config"].model:
                cfg.model.transreid.merge_from_other_cfg(checkpoint["config"].model.transreid)
        else:
            print("Could not load config from file {}".format(cfg.model.load_weights))

    if display_diff:
        display_config_diff(cfg, default_cfg_copy)

    cfg.data.save_dir = os.path.join(cfg.data.save_dir, str(cfg.project.job_id))
    os.makedirs(cfg.data.save_dir)
    print(
        "Save dir created at {}".format(
            os.path.join(pathlib.Path().resolve(), cfg.data.save_dir)
        )
    )
    return cfg


def build_torchreid_model_engine(cfg):
    if cfg.project.debug_mode:
        torch.autograd.set_detect_anomaly(True)
    logger = Logger(cfg)
    writer = Writer(cfg)
    set_random_seed(cfg.train.seed)
    print("Show configuration\n{}\n".format(cfg))
    print(f"Effective batch size: {cfg.train.batch_size} (batch size) * {cfg.train.accumulation_steps} (accumulation steps) = {cfg.train.batch_size * cfg.train.accumulation_steps}")
    print("Collecting env info ...")
    if cfg.use_gpu:
        torch.backends.cudnn.benchmark = True

    # Check if this is test-only mode (zeroshot evaluation)
    test_only = getattr(cfg.test, 'evaluate', False)
    datamanager = build_datamanager(cfg, test_only=test_only)

    engine_state = EngineState(cfg.train.start_epoch, cfg.train.max_epoch)
    writer.init_engine_state(engine_state, cfg.model.kpr.masks.parts_num)
    print("Building model: {}".format(cfg.model.name))
    model = build_model(cfg, datamanager.num_train_pids, datamanager.train_loader.dataset.cam_num, datamanager.train_loader.dataset.view)
    logger.add_model(model)
    optimizer = torchreid.optim.build_optimizer(model, **optimizer_kwargs(cfg))
    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer, **lr_scheduler_kwargs(cfg)
    )
    if cfg.model.resume and check_isfile(cfg.model.resume):
        cfg.train.start_epoch = resume_from_checkpoint(
            cfg.model.resume, model, optimizer=optimizer, scheduler=scheduler
        )
    print("Building {}-engine for {}-reid".format(cfg.loss.name, cfg.data.type))
    engine = build_engine(
        cfg, datamanager, model, optimizer, scheduler, writer, engine_state
    )
    return engine, model


def build_model(cfg, num_train_pids=1, cam_num=0, view=0):
    model_name = cfg.model.name
    if model_name in swin_timm_models:
        model = swin_timm_models[model_name](
            config=cfg,
            cam_num=cam_num,
            view=view,
            num_classes=num_train_pids
        )
    else:
        model = torchreid.models.build_model(
            name=cfg.model.name,
            num_classes=num_train_pids,
            loss=cfg.loss.name,
            pretrained=cfg.model.pretrained,
            use_gpu=cfg.use_gpu,
            cam_num=cam_num,
            view=view,
            config=cfg,
        )
    cfg.model.kpr.spatial_feature_shape = model.spatial_feature_shape
    if cfg.model.compute_complexity:
        num_params, flops = compute_model_complexity(model, cfg)
        print("Model complexity: params={:,} flops={:,}".format(num_params, flops))
    if cfg.model.load_weights and check_isfile(cfg.model.load_weights):
        load_pretrained_weights(model, cfg.model.load_weights)
    if cfg.use_gpu:
        model = nn.DataParallel(model).cuda()
        print('Model moved to multi GPU')
        # if torch.cuda.device_count() > 1:
        #     # 분산 학습 초기화
        #     if not torch.distributed.is_initialized():
        #         torch.distributed.init_process_group(backend='nccl')
        #     local_rank = torch.distributed.get_rank()
        #     torch.cuda.set_device(local_rank)
        #     model = model.cuda()
        #     model = torch.nn.parallel.DistributedDataParallel(
        #         model, device_ids=[local_rank], output_device=local_rank
        #     )
        #     print(f'Model moved to DistributedDataParallel on GPU {local_rank}')
        # else:
        #     model = model.cuda()
        #     print('Model moved to single GPU')
    
    return model
