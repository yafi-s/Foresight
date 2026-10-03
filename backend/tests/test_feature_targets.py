"""Exercise the actual pure feature function without starting provider/DB code."""
import ast
import logging
from pathlib import Path
from typing import Tuple,List
import unittest
import numpy as np
import pandas as pd


class FeatureTargetsTests(unittest.TestCase):
    def test_future_labels_use_original_time_axis_after_feature_gaps(self):
        source=Path(__file__).resolve().parents[1]/'data_loader.py'
        module=ast.parse(source.read_text(encoding='utf8'))
        function=next(node for node in module.body if isinstance(node,ast.FunctionDef) and node.name=='prepare_features')
        namespace={'np':np,'pd':pd,'logger':logging.getLogger('test'),'Tuple':Tuple,'List':List,'PREDICTION_DAYS':5}
        exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),namespace)
        t=np.arange(100,dtype=float)
        close=100+t+np.sin(t)
        frame=pd.DataFrame({'Close':close,'High':close+1,'Low':close-1,'Volume':1000+t*10})
        frame.loc[80,'Volume']=np.nan
        x,y,_=namespace['prepare_features'](frame)
        self.assertNotIn(80,x.index)
        self.assertAlmostEqual(y.loc[76,'Future_Return_5'],close[81]/close[76]-1)
        self.assertNotAlmostEqual(y.loc[76,'Future_Return_5'],close[82]/close[76]-1)


if __name__=='__main__':
    unittest.main()
