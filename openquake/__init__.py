#
# Empty file necessary for python to recognise directory as package
#

import IO
reload(IO)

import config
reload(config)

from IO import parse_hazard_curves, parse_hazard_map

from config import OQ_Params
