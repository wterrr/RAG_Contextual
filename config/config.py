import os
import yaml
from easydict import EasyDict as edict

class YamlParser(edict):
    def __init__(self, cfg_dict=None, cfg_file=None):
        if cfg_dict is None:
            cfg_dict = {}
        
        if cfg_file is not None:
            assert(os.path.isfile(cfg_file))
            with open(cfg_file, 'r') as f:
                cfg_dict.update(yaml.load(f.read(), Loader=yaml.FullLoader))
        super(YamlParser, self).__init__(cfg_dict)
        
    def merge_from_dict(self, cfg_dict):
        self.update(cfg_dict)
        
    def merge_from_file(self, cfg_file):
        with open(cfg_file, 'r') as f:
            self.update(yaml.load(f.read(), Loader=yaml.FullLoader))
            
def get_config(cfg_file=None):
    return YamlParser(cfg_file=cfg_file)