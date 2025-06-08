import random
from torch.utils.data.dataloader import default_collate

def custom_collate_fn(batch, aug_map, aug_ratio=0.1):
    batch_size = len(batch)
    n_aug = int(round(batch_size * aug_ratio))
    if n_aug > 0:
        aug_indices = random.sample(range(batch_size), n_aug)
        for i in aug_indices:
            img_path = batch[i]['img_path']
            if img_path in aug_map and aug_map[img_path]:
                batch[i] = batch[i].copy()
                batch[i]['img_path'] = random.choice(aug_map[img_path])
    return default_collate(batch)
