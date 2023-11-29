from enum import Enum

class WindowType(Enum):
    Variable = 'Variable_Length'
    Fixed = 'Fixed_Length'

class Models(Enum):
    CNN = 'CNN'
    LSTM = 'LSTM'

class Subsets(Enum):
    FD001 = 'FD001'
    FD002 = 'FD002'
    FD003 = 'FD003'
    FD004 = 'FD004'
    
class GM(Enum):
    gtw = 'gtw'
    gow = 'gow'
    wog = 'wog'
    zeros = 0
    ones = 1
    vgm = 'vgm'
    mgm = 'mgm'
