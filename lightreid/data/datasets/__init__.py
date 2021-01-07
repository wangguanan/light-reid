from .market1501 import Market1501
from .dukemtmcreid import DukeMTMCreID
from .msmt17 import MSMT17
from .cuhk03 import CUHK03
from .wildtrack import WildTrackCrop
from .rap import RAP
from .njust365 import NJUST365, NJUST365SPR, NJUST365WIN
from .airportalert import AirportAlert
from .prid import PRID
from .occluded_reid import OccludedReID
from .partial_ilids import PartialILIDS
from .partial_reid import PartialReID

from .build import build_train_dataset, build_test_dataset


__all__ = [
    'Market1501', 'DukeMTMCreID', 'MSMT17', 'CUHK03', 'WildTrackCrop', 'RAP',
    'NJUST365', 'NJUST365WIN', 'NJUST365SPR', 'AirportAlert', 'PRID',
    'OccludedReID', 'PartialILIDS', 'PartialReID',
    'build_train_dataset', 'build_test_dataset']
