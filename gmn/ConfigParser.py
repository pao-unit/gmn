# Python distribution modules
import configparser

# Local modules
from .Parameters import Parameters

# Typed accessors for ConfigMap rows. Bound to the class, not an
# instance : called as Getter( config, section, key ).
GetStr   = configparser.ConfigParser.get
GetInt   = configparser.ConfigParser.getint
GetFloat = configparser.ConfigParser.getfloat
GetBool  = configparser.ConfigParser.getboolean

# Config file key -> Parameters attribute map. One row per key :
#   ( section, key, Parameters attribute, typed accessor )
# ReadConfig() overlays only the keys present in the file, so this is
# also the authority on which sections & keys are recognised. Adding a
# parameter is one row here plus one default in Parameters.__init__.
ConfigMap = (
    ( 'GMN',     'mode',              'mode',              GetStr   ),
    ( 'GMN',     'predictionStart',   'predictionStart',   GetInt   ),
    ( 'GMN',     'predictionLength',  'predictionLength',  GetInt   ),
    ( 'GMN',     'outPath',           'outPath',           GetStr   ),
    ( 'GMN',     'dataOutFile',       'dataOutFile',       GetStr   ),
    ( 'GMN',     'showPlot',          'showPlot',          GetBool  ),
    ( 'GMN',     'plotType',          'plotType',          GetStr   ),
    ( 'GMN',     'plotColumns',       'plotColumns',       GetStr   ),
    ( 'GMN',     'plotFile',          'plotFile',          GetStr   ),
    ( 'GMN',     'backend',           'backend',           GetStr   ),
    ( 'GMN',     'kernel',            'kernel',            GetBool  ),

    ( 'Network', 'name',              'networkName',       GetStr   ),
    ( 'Network', 'targetNode',        'targetNode',        GetStr   ),
    ( 'Network', 'file',              'networkFile',       GetStr   ),
    ( 'Network', 'data',              'networkData',       GetStr   ),

    ( 'Node',    'info',              'nodeInfo',          GetStr   ),
    ( 'Node',    'function',          'function',          GetStr   ),
    ( 'Node',    'data',              'nodeData',          GetStr   ),
    ( 'Node',    'configPath',        'nodeConfigPath',    GetStr   ),

    ( 'EDM',     'lib',               'lib',               GetStr   ),
    ( 'EDM',     'pred',              'pred',              GetStr   ),
    ( 'EDM',     'E',                 'E',                 GetInt   ),
    ( 'EDM',     'Tp',                'Tp',                GetInt   ),
    ( 'EDM',     'knn',               'knn',               GetInt   ),
    ( 'EDM',     'tau',               'tau',               GetInt   ),
    ( 'EDM',     'theta',             'theta',             GetFloat ),
    ( 'EDM',     'exclusionRadius',   'exclusionRadius',   GetInt   ),
    ( 'EDM',     'columns',           'columns',           GetStr   ),
    ( 'EDM',     'target',            'target',            GetStr   ),
    ( 'EDM',     'solver',            'solver',            GetStr   ),
    ( 'EDM',     'embedded',          'embedded',          GetBool  ),
    ( 'EDM',     'validLib',          'validLib',          GetStr   ),
    ( 'EDM',     'generateSteps',     'generateSteps',     GetInt   ),
    ( 'EDM',     'libSizes',          'libSizes',          GetStr   ),
    ( 'EDM',     'sample',            'sample',            GetInt   ),
    ( 'EDM',     'random',            'random',            GetBool  ),
    ( 'EDM',     'includeData',       'includeData',       GetBool  ),
    ( 'EDM',     'seed',              'seed',              GetInt   ),

    ( 'Scale',   'factor',            'factor',            GetFloat ),
    ( 'Scale',   'offset',            'offset',            GetFloat ),
)

#------------------------------------------------------------------------
#------------------------------------------------------------------------
def ReadConfig( args, configurationFile = None ):
    '''Read configuration file, parse into Parameters object.

    A partial config file is valid : only the keys present are read,
    every other Parameters attribute keeps the default assigned in
    Parameters.__init__. Absent sections are ignored, so a config file
    may specify [EDM] alone. Unrecognised sections & keys are reported
    so that a misspelled key is not silently defaulted.
    '''

    configFile = None

    if not configurationFile:
        configFile = args.configFile

    else :
        configFile = configurationFile

    if not configFile:
        raise RuntimeError( "ReadConfig(): configFile not specified." )

    config = configparser.ConfigParser()

    if not config.read( configFile ):
        raise RuntimeError( 'ReadConfig(): failed to read ' + configFile )

    # Overlay config file values onto the Parameters defaults
    param       = Parameters()
    validLibSet = False

    for section, key, attribute, Getter in ConfigMap :

        if config.has_option( section, key ) :
            setattr( param, attribute, Getter( config, section, key ) )

            if attribute == 'validLib' :
                validLibSet = True

    # Convert validLib to list of int if read from the config file and
    # not empty. If not read it is already the default empty list.
    if validLibSet :

        if len( param.validLib ) == 0 or param.validLib.isspace():
            param.validLib = []

        else:
            param.validLib = [ int(x) for x in param.validLib.split() ]

    ReportUnknown( configFile, config )

    return param

#------------------------------------------------------------------------
#------------------------------------------------------------------------
def ReportUnknown( configFile, config ):
    '''Print config file sections & keys that are not in ConfigMap.

    Since omitted keys now fall back to defaults, a misspelled key would
    otherwise be silently ignored instead of raising as it once did.
    '''

    knownSections = { row[0] for row in ConfigMap }

    # configparser lowercases option names : match ConfigMap the same way
    knownKeys = { ( row[0], config.optionxform( row[1] ) )
                  for row in ConfigMap }

    for section in config.sections() :

        if section not in knownSections :
            print( 'ReadConfig():', configFile, ': unknown section [' +
                   section + ']', flush = True )
            continue

        for key in config.options( section ) :

            if ( section, key ) not in knownKeys :
                print( 'ReadConfig():', configFile, ': unknown key', key,
                       'in [' + section + ']', flush = True )
