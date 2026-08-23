## GMN Methods

### <function> GMN </function> 
**Description** :  
Class object for Generative Manifold Networks (GMN).

**Constructor Signature**
```python
class GMN:
    def __init__( self,
                  args        = None,  parameters = None,
                  configFile  = None,  configDir  = None,
                  outputFile  = None,  cores      = 4,
                  backend     = None,  chunks     = None,
                  kdWorkers   = None,  kernel     = None,
                  plot        = False, plotType   = 'state',
                  plotColumns = [],    plotFile   = None,
                  figureSize  = [8,8], verbose    = False, debug = False ):
```

Full configuration is performed by instantiating GMN with a config file: `gmn = GMN( configFile = 'myConfig.cfg' )`. It can also be configured passing in `args` from [gmn.CLI_Parser.ParseCmdLine()](https://github.com/NonlinearDynamicsDSU/gmn/blob/master/gmn/CLI_Parser.py) and `parameters` from [gmn.ConfigParser.ReadConfig()](https://github.com/NonlinearDynamicsDSU/gmn/blob/master/gmn/ConfigParser.py). This is done by the command-line-interface (CLI) program [Run.py](https://github.com/NonlinearDynamicsDSU/gmn/blob/master/apps/Run.py). [RunNoConfig.py](https://github.com/NonlinearDynamicsDSU/gmn/blob/master/apps/RunNoConfig.py) does neither: it has its own argument parser and populates a `Parameters` object directly from the command line, since its purpose is to run without a config file.

If `args` is `None` `GMN.__init__` function arguments are used to populate args. 

If `parameters` is `None` the `Parameters` object is read from the config file named by `args.configFile`, via `ConfigParser.ReadConfig()`. Command line values are not merged into it: they remain on `args`.

A config file may specify any subset of the recognized keys. Keys omitted from the file keep the default assigned in `Parameters.__init__`.

`backend` and `kernel` default to `None` so that a config file can set them. Each is resolved as: an explicit constructor or CLI value, otherwise the config file `[GMN]` key, otherwise the `Parameters` default (`'serial'`, `True`).

| Parameter | Type | Default | Purpose |
| --------- | ---- | ------- | ------- |
| args        | argparse ArgumentParser | None | command line arguments
| parameters  | gmn Parameter object    | None | GMN parameters
| configFile  | string | None | Configuration file for Network / Nodes |
| configDir   | string | None | Path to directory of configuration file(s) |
| outputFile  | string | None | Generated data output file : .csv or .feather |
| cores       | int    | 4    | Number of CPU processor cores |
| backend     | string | None | 'serial' or 'parallel' |
| chunks      | int    | None | parallel chunk size |
| kdWorkers   | int    | None | thread workers in kdTree.query() : pyEDM only |
| kernel      | bool   | None | True - float32 Simplex kernel : False - pyEDM node functions |
| plot        | bool   | False| Logical to plot time series results |
| plotType    | string | 'state' | 'time' or 'state' plot |
| plotColumns | list   | []   | List of columns to plot |
| plotFile    | string | None | File for plot results |
| figureSize  | list   | [8,8]| Plot figure size in inches |
| verbose     | bool   | False| Logical for verbose output |
| debug       | bool   | False| Logical for debug output, enables faulthandler |


**Returns**  :  
`GMN` class object.  The object is initialized to create the `GMN.Network` class object, all `Node` class objects of the network including input data, and the `GMN.DataOut` pandas DataFrame. 

**Example** :  
```python
# Initalize GMN object with default.cfg
import gmn
G = gmn.GMN( configFile = 'config/default.cfg' )
```

---

### <function> GMN.Generate </function> 

**Description**  :   
Execute GMN forecast loop for `predictionLength` steps calling the Generate() method of each Network Node. Parameters `mode` must be `Generate`.

**Returns**  :  
Populates the `GMN.DataOut` pandas DataFrame. 

**Notes** :
If `args.outputFile`, or `parameters.dataOutFile`: write `DataOut` as a .csv or .feather file according to the `dataOutFile` file extension.

if args.Plot or args.StatePlot or parameters.showPlot or parameters.plotFile: call GMN.Plot()

**Example** :  
```python
G.Generate()
```
---

### <function> GMN.Forecast </function> 

**Description**  :   
Execute GMN `Forecast()` method of each Network Node. Parameters `mode` must not be `Generate`. Presumes Parameters `lib` and `pred` are specified in the config file. Does not generate data, but makes predictions over the `pred` indices based on the `lib` state-space.

**Returns**  :  
Populates the `GMN.DataOut` pandas DataFrame. 

**Notes** :  
If `args.outputFile`, or `parameters.dataOutFile`: write `DataOut` as a .csv or .feather file according to the `dataOutFile` file extension.

if args.Plot or args.StatePlot or parameters.showPlot or parameters.plotFile: call GMN.Plot()

**Example** :  
```python
G.Forecast()
```
---

### <function> GMN.Plot </function>

**Description**  :   
Plot generated time series (args.Plot = True, or Parameters.plotType is 'time') or time series and 2-D state-space plots (args.StatePlot = True, or Parameters.plotType is 'state').

**Returns**  :  
pyplot image 

---

## GMN Attributes

### GMN.DataOut

**Description**  :   
pandas DataFrame of generated data.

### GMN.Parameters

**Description**  :  
Python object of Parameters class.
