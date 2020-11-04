from .data import build_datamanager
from .models import build_model

def build(config):

    datamanager = build_datamanager(**config.data)
    build_model = build_model(**config.model)

