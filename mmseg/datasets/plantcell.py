# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class PlantCellDataset(BaseSegDataset):
    """
	Plant Cell dataset.
    """
    METAINFO = dict(
        classes = ('background', 'Chloroplast', 'Mitochondria', 'Vacuole', 'Nucleus'),
        palette = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128]])

    def __init__(self,
                 img_suffix='.tif',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
