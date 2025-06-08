from __future__ import division, print_function, absolute_import

from torchreid import metrics
from torchreid.losses import CrossEntropyLoss

from ..engine import Engine


class ImageSoftmaxEngineAccumulation(Engine):
    r"""Softmax-loss engine for image-reid with gradient accumulation.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        scheduler (LRScheduler, optional): if None, no learning rate decay will be performed.
        use_gpu (bool, optional): use gpu. Default is True.
        label_smooth (bool, optional): use label smoothing regularizer. Default is True.
        accumulation_steps (int, optional): number of steps to accumulate gradients. Default is 1.

    Examples::
        
        import torchreid
        datamanager = torchreid.data.ImageDataManager(
            root='path/to/reid-data',
            sources='market1501',
            height=256,
            width=128,
            combineall=False,
            batch_size=32
        )
        model = torchreid.models.build_model(
            name='resnet50',
            num_classes=datamanager.num_train_pids,
            loss='softmax'
        )
        model = model.cuda()
        optimizer = torchreid.optim.build_optimizer(
            model, optim='adam', lr=0.0003
        )
        scheduler = torchreid.optim.build_lr_scheduler(
            optimizer,
            lr_scheduler='single_step',
            stepsize=20
        )
        engine = torchreid.engine.ImageSoftmaxEngineAccumulation(
            datamanager, model, optimizer, scheduler=scheduler, accumulation_steps=4
        )
        engine.run(
            max_epoch=60,
            save_dir='log/resnet50-softmax-market1501'
        )
    """

    def __init__(
        self,
        datamanager,
        model,
        optimizer,
        writer,
        engine_state,
        scheduler=None,
        use_gpu=True,
        label_smooth=True,
        save_model_flag=False,
        accumulation_steps=1
    ):
        super(ImageSoftmaxEngineAccumulation, self).__init__(datamanager, writer, engine_state, use_gpu, save_model_flag)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.accumulation_steps = accumulation_steps
        self.register_model('model', model, optimizer, scheduler)

        self.criterion = CrossEntropyLoss(
            label_smooth=label_smooth
        )
        self.step_count = 0

    def forward_backward(self, data):
        imgs, pids = self.parse_data_for_train(data)

        if self.use_gpu:
            imgs = imgs.cuda()
            pids = pids.cuda()

        outputs = self.model(imgs)
        loss = self.compute_loss(self.criterion, outputs, pids)

        loss.backward()

        self.step_count += 1
        if self.step_count % self.accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        loss_summary = {
            'loss': loss.item(),
            'acc': metrics.accuracy(outputs, pids)[0].item()
        }

        return loss, {'glb_ft': loss_summary}
