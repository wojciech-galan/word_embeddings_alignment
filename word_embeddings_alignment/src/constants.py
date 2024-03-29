import os
import json

_this_dir = os.path.dirname(__file__)
PACKAGE_DIR = _this_dir.rsplit(os.path.sep, 2)[0]
_conf_path = os.path.join(PACKAGE_DIR, 'conf.json')
CONF = json.load(open(_conf_path))
PROT_VEC_CSV = os.path.join(PACKAGE_DIR, CONF['prot_dir'], CONF['prot_file'])
PROT_VEC_PICKLE = os.path.join(PACKAGE_DIR, CONF['prot_dir'], CONF['prot_file_serialized'])