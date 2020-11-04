"""
@author:    wangguanan
@contact:   guan.wang0706@gmail.com
"""

import os, copy
from .reid_samples import ReIDSamples


class PRID(ReIDSamples):
    """PRID.
    Reference:
        Person Re-Identification by Descriptive and Discriminative Classification
        Martin Hirzer, Csaba Beleznai, Peter M. Roth, and Horst Bischof
        In Proc. Scandinavian Conference on Image Analysis, 2011
    ReadMe:
        Camera view A shows 385 persons, camera view B shows 749 persons. The first 200
        persons appear in both camera views, i.e., person 0001 of view A corresponds to
        person 0001 of view B, person 0002 of view A corresponds to person 0002 of view
        B, and so on. The remaining persons in each camera view (i.e., person 0201 to
        0385 in view A and person 0201 to 0749 in view B) complete the gallery set of
        the corresponding view. Hence, a typical evaluation consists of searching the
        200 first persons of one camera view in all persons of the other view. This means
        that there are two possible evalutaion procedures, either the probe set is drawn
        from view A and the gallery set is drawn from view B (A to B, used in [1]), or
        vice versa (B to A).
    Dataset statistics:
        - identities: 385
        - images: 2.
    Args:
        data_path(str): path to PRID dataset
        combineall(bool): combine train and test sets as train set if True
    """

    def __init__(self, data_path, combineall=False, download=False):

        train = []
        for idx in range(1, 201):
            cama_path = os.path.join(data_path, 'multi_shot/cam_a/person_{0:04d}'.format(idx))
            camb_path = os.path.join(data_path, 'multi_shot/cam_b/person_{0:04d}'.format(idx))
            _, _, files_a = self.os_walk(cama_path)
            for file_a in files_a:
                train.append([os.path.join(cama_path, file_a), idx, 0])
            _, _, files_b = self.os_walk(camb_path)
            for file_b in files_b:
                train.append([os.path.join(camb_path, file_b), idx, 0])

        query = None
        gallery = None

        # init
        super(PRID, self).__init__(train, query, gallery)


if __name__ == '__main__':
    AirportAlert('/data/datasets/airport')