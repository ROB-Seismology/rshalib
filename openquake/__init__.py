#
# Empty file necessary for python to recognise directory as package
#

import IO
reload(IO)

import config
## Reloading config may in some cases result in the following error:
## super(OQ_Params, self).__init__(ini_filespec, configspec=configspec, write_empty_values=True, list_values=False)
## TypeError: super(type, obj): obj must be an instance or subtype of type
reload(config)

from IO import parse_hazard_curves, parse_hazard_map, parse_uh_spectra, parse_disaggregation

from config import OQ_Params
