from river.compat import River2SKLTransformer
from sail.models.river.base import RiverMixin


class BaseRiverTransformer(River2SKLTransformer, RiverMixin):
    def __init__(self, *args, **Kwargs):
        super(BaseRiverTransformer, self).__init__(*args, **Kwargs)
