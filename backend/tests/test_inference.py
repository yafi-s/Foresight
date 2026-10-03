"""Exercise actual serving functions without provider/DB/TensorFlow side effects."""
import ast
from datetime import datetime
import logging
from pathlib import Path
import tempfile
from typing import Dict,List,Tuple
import unittest
import numpy as np
import pandas as pd
from artifact_protocol import validate_bundle,write_bundle_metadata
from causal import prepare_split,prepare_inference,prices_from_returns,estimated_weekday_dates


ROOT=Path(__file__).resolve().parents[1]


def function(filename,name,namespace):
    module=ast.parse((ROOT/filename).read_text(encoding='utf8'))
    node=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name==name)
    node.decorator_list=[]
    # TensorFlow annotations are irrelevant to this actual function's execution.
    node.returns=None
    for argument in node.args.args+node.args.kwonlyargs:
        argument.annotation=None
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(ROOT/filename),'exec'),namespace)
    return namespace[name]


class IdentityScaler:
    def transform(self,x): return np.asarray(x)
    def inverse_transform(self,x): return np.asarray(x)


class InferenceTests(unittest.TestCase):
    def frame(self):
        t=np.arange(100,dtype=float)
        close=100+t+np.sin(t)
        return pd.DataFrame({'Close':close,'High':close+1,'Low':close-1,'Volume':1000+t*10},
                            index=pd.bdate_range(end='2026-10-02',periods=100))

    def features(self):
        return function('data_loader.py','prepare_features',{'np':np,'pd':pd,'Tuple':Tuple,'List':List,
                        'PREDICTION_DAYS':5,'logger':logging.getLogger('test')})

    def test_latest_observations_retained_and_close_channel_aligned(self):
        df=self.frame()
        prepare=self.features()
        training,targets,_=prepare(df.copy())
        inference,labels,_=prepare(df.copy(),include_targets=False)
        self.assertEqual(training.index[-1],df.index[-1])
        self.assertTrue(targets.iloc[-5:].isna().any(axis=1).all())
        self.assertEqual(inference.index[-1],df.index[-1])
        self.assertEqual(labels.shape[1],0)
        closes=df.loc[inference.index,'Close'].to_numpy()
        sequence=prepare_inference(inference.to_numpy(),closes,IdentityScaler())
        np.testing.assert_allclose(sequence[0,:,-1],df.Close.iloc[-30:].to_numpy())
        self.assertNotEqual(sequence[0,0,-1],sequence[0,-1,-1])

    def test_missing_future_labels_cannot_remove_training_history(self):
        t=np.arange(300,dtype=float)
        close=100+t+np.sin(t)
        frame=pd.DataFrame({'Close':close,'High':close+1,'Low':close-1,'Volume':1000+t*10})
        frame.loc[120,'Close']=np.nan
        prepare=self.features()
        x,y,_=prepare(frame.copy())
        observed,_,_=prepare(frame.copy(),include_targets=False)
        pd.testing.assert_frame_equal(x,observed)
        split=prepare_split(x.to_numpy(),y.to_numpy(),frame.loc[x.index,'Close'].to_numpy(),x.index.to_numpy())
        # Future missing prices affect label eligibility, not prior input rows.
        position=140
        sample=np.flatnonzero(split.train_origins==position)[0]
        history=x.loc[:position]
        inference=prepare_inference(history.to_numpy(),frame.loc[history.index,'Close'].to_numpy(),split.feature_scaler)
        np.testing.assert_allclose(inference[0],split.X_train[sample])

    def test_actual_prediction_uses_origin_prices_and_weekday_estimates(self):
        fake=type('Model',(),{'predict':lambda self,x,verbose=0:np.array([[.1,.2,.3,.4,.5]])})()
        predict=function('model.py','predict_next_week',{'np':np,'PREDICTION_DAYS':5,
                         'prices_from_returns':prices_from_returns,'estimated_weekday_dates':estimated_weekday_dates})
        result=predict(fake,IdentityScaler(),np.zeros((1,30,9)),100.,origin_date='2026-10-02')
        np.testing.assert_allclose(result['predictions'],[110,120,130,140,150])
        self.assertEqual(result['dates'],['2026-10-05','2026-10-06','2026-10-07','2026-10-08','2026-10-09'])
        self.assertIn('holidays not modeled',result['date_basis'])

    def test_actual_api_prediction_uses_target_free_latest_input(self):
        import os
        df=self.frame()
        recorded={}
        scaler=IdentityScaler()
        class FakeModel:
            def predict(self,x,verbose=0):
                recorded['input']=x.copy()
                return np.array([[.1,.2,.3,.4,.5]])
        predictor=function('model.py','predict_next_week',{'np':np,'PREDICTION_DAYS':5,
                           'prices_from_returns':prices_from_returns,'estimated_weekday_dates':estimated_weekday_dates})
        session=type('Session',(),{'add':lambda self,x:None,'commit':lambda self:None})()
        prediction=type('Prediction',(),{'__init__':lambda self,**kwargs:None})
        class HttpError(Exception): pass
        ns={'check_model_exists':lambda _:True,'validate_bundle':lambda *_:None,'MODEL_DIR':'.',
            'load_stock_data':lambda _:df,'prepare_features':self.features(),'prepare_inference':prepare_inference,
            'logger':logging.getLogger('test'),'os':type('OS',(),{'path':type('Path',(),{'join':os.path.join,'exists':lambda _:True})}),
            'joblib':type('Joblib',(),{'load':lambda _:scaler}),'SEQUENCE_LENGTH':30,
            'load_trained_model':lambda _:FakeModel(),'predict_next_week':predictor,'Prediction':prediction,
            'datetime':datetime,'HTTPException':HttpError,'Depends':lambda _:None,'get_session':lambda:None}
        api=function('main.py','predict',ns)
        result=api('TEST',session)
        np.testing.assert_allclose(recorded['input'][0,:,-1],df.Close.iloc[-30:].to_numpy())
        self.assertEqual(result['forecastOrigin'],'2026-10-02')
        self.assertEqual(result['lastActualClose'],df.Close.iloc[-1])

    def test_bundle_rejects_legacy_and_mixed_artifacts(self):
        with tempfile.TemporaryDirectory() as work:
            with self.assertRaisesRegex(ValueError,'Legacy'):
                validate_bundle(work,'TEST')
            for suffix in ('_best.keras','_best.h5','_feature_scaler.joblib','_target_scaler.joblib'):
                (Path(work)/f'TEST{suffix}').write_bytes(suffix.encode())
            write_bundle_metadata(work,'TEST',['feature'])
            self.assertEqual(validate_bundle(work,'TEST')['protocol'],'causal-v2')
            (Path(work)/'TEST_feature_scaler.joblib').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'changed'):
                validate_bundle(work,'TEST')


if __name__=='__main__':
    unittest.main()
