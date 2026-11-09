"""CPU deployment of the frozen plain TCN; no labels or reference CSV inputs."""
from collections import deque
import json
from pathlib import Path
import time
import numpy as np
from .ml_data import CLASSES,FEATURE_NAMES,sha256

TITLES={'squats':'Squats','pushups':'Push-ups','lunges':'Lunges','bicep_curls':'Bicep curls',
        'other_exercise':'Other exercise','non_activity':'Rest / setup'}


class ExerciseClassifier:
    def __init__(self,assets=None):
        assets=Path(assets or Path(__file__).resolve().parents[2]/'assets')
        path=assets/'exercise_plain_tcn.onnx';meta_path=assets/'exercise_plain_tcn.json'
        if not path.is_file() or not meta_path.is_file():
            raise ValueError('The trained AI model is missing from assets/. Use the project AI demo setup.')
        self.metadata=json.loads(meta_path.read_text(encoding='utf-8'))
        if self.metadata['classes']!=list(CLASSES) or self.metadata['source_feature_names']!=list(FEATURE_NAMES):
            raise ValueError('AI model and feature schema differ.')
        if sha256(path)!=self.metadata['onnx_sha256']:
            raise ValueError('AI model checksum differs from its metadata.')
        try:import onnxruntime as ort
        except ImportError as error:
            raise ValueError('AI inference dependency missing. Install requirements-ai.txt in the project environment.') from error
        options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
        self.inference_s=0.;self.calls=0

    def predict(self,features,usable):
        features=np.asarray(features,np.float32);usable=np.asarray(usable,bool)
        if features.shape!=(45,239) or usable.shape!=(45,) or not np.isfinite(features).all():
            raise ValueError('AI prediction requires 45 finite feature observations and validity masks.')
        start=time.perf_counter()
        logits=self.session.run(None,{'features':features[None], 'usable':usable[None]})[0][0]
        self.inference_s+=time.perf_counter()-start;self.calls+=1
        probabilities=np.exp(logits-logits.max());probabilities/=probabilities.sum()
        return probabilities.astype(np.float32)


class RecognitionStream:
    """Causal 3-second windows, eight-observation stride, two confirming votes.

    Stabilization is a demonstration policy, not a calibrated uncertainty rule.
    No retroactive relabeling occurs; warm-up and transitions stay inspectable.
    """
    def __init__(self,classifier):
        self.classifier=classifier;self.buffer=deque(maxlen=45)
        self.observations=0;self.last_valid=None;self.previous_time=None
        self.stable=None;self.candidate=None;self.votes=0;self.raw=None;self.probabilities=None
        self.coverage=0.;self.window_start=None;self.window_end=None

    def reset(self):
        self.buffer.clear();self.observations=0
        self.stable=self.candidate=self.raw=None;self.votes=0;self.probabilities=None
        self.coverage=0.;self.window_start=self.window_end=None

    def update(self,features,usable,timestamp):
        if self.previous_time is not None and timestamp-self.previous_time>1.6/15:
            self.reset()
        self.previous_time=timestamp
        if usable:self.last_valid=timestamp
        elif self.last_valid is None or timestamp-self.last_valid>.4:
            self.reset()
        self.buffer.append((np.asarray(features,np.float32),bool(usable),float(timestamp)))
        self.observations+=1;updated=False
        if len(self.buffer)==45 and (self.observations-45)%8==0:
            updated=True;self.coverage=sum(item[1] for item in self.buffer)/45
            self.window_start=self.buffer[0][2];self.window_end=timestamp
            if self.coverage>=.8:
                self.probabilities=self.classifier.predict(np.stack([x[0] for x in self.buffer]),
                    np.array([x[1] for x in self.buffer],bool))
                self.raw=CLASSES[int(self.probabilities.argmax())]
                if self.raw==self.candidate:self.votes+=1
                else:self.candidate=self.raw;self.votes=1
                if self.votes>=2:self.stable=self.raw
            else:
                self.raw=self.stable=self.candidate=None;self.votes=0;self.probabilities=None
        status='tracking_limited' if not usable else 'warming_up' if len(self.buffer)<45 else (
            'tracking_limited' if self.probabilities is None else 'confirming' if self.raw!=self.stable else 'ready')
        return dict(raw_class=self.raw,stable_class=self.stable,
            raw_score=float(self.probabilities.max()) if self.probabilities is not None else None,
            status=status,window_coverage=self.coverage,updated=updated,
            window_start_s=self.window_start,window_end_s=self.window_end)
