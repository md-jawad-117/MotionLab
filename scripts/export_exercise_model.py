"""Export the frozen plain TCN with its exact source selection and normalization."""
from pathlib import Path
import argparse
import hashlib
import io
import importlib.util
import json
import zipfile
import numpy as np
import torch
from torch import nn
import onnx
import onnxruntime as ort

HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--project',type=Path,default=HERE.parent)
parser.add_argument('--output',type=Path)
args=parser.parse_args()
project=args.project.resolve()
spec=importlib.util.spec_from_file_location('trained_architecture',project/'ml/evaluate_motionlite.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
checkpoint=(project/'evidence/checkpoints/plain_tcn/best.pt').read_bytes()
assert hashlib.sha256(checkpoint).hexdigest()==module.FROZEN_CHECKPOINTS['plain_tcn']
state=torch.load(io.BytesIO(checkpoint),map_location='cpu',weights_only=True)
model=module.PlainTCN().eval();model.load_state_dict(state['model'])
protocol=state['protocol']


class DeploymentModel(nn.Module):
    def __init__(self):
        super().__init__();self.model=model
        self.register_buffer('columns',torch.tensor(module.SOURCE_COLUMNS,dtype=torch.int64))
        self.register_buffer('mean',torch.tensor(protocol['normalization']['mean'],dtype=torch.float32))
        self.register_buffer('std',torch.tensor(protocol['normalization']['std'],dtype=torch.float32))
    def forward(self,features,usable):
        values=features.index_select(2,self.columns)
        mask=torch.cat((values[:,:,60:72].repeat_interleave(2,dim=2),
            values[:,:,72:84].repeat_interleave(2,dim=2)),dim=2)>.5
        normalized=torch.where(mask,((values[:,:,:48]-self.mean)/self.std).clamp(-8,8),
            torch.zeros_like(values[:,:,:48]))
        values=torch.cat((normalized,values[:,:,48:]),dim=2)
        return self.model(values,usable)


torch.set_num_threads(2)
deployment=DeploymentModel().eval()
out=args.output or project/'assets';out.mkdir(parents=True,exist_ok=True)
path=out/'exercise_plain_tcn.onnx'
torch.onnx.export(deployment,(torch.zeros(1,45,239),torch.ones(1,45,dtype=torch.bool)),str(path),
    input_names=['features','usable'],output_names=['logits'],opset_version=17,dynamo=False,
    do_constant_folding=True)
onnx.checker.check_model(onnx.load(path))
options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
rng=np.random.default_rng(42);errors=[]
for i in range(20):
    features=rng.normal(0,.5,(1,45,239)).astype(np.float32)
    features[:,:,132:165]=rng.uniform(0,1,(1,45,33))
    features[:,:,165:231]=rng.integers(0,2,(1,45,66)).astype(np.float32)
    features[:,:,231:235]=rng.uniform(0,1,(1,45,4))
    features[:,:,235:239]=rng.integers(0,2,(1,45,4))
    usable=(rng.random((1,45))>.1)
    with torch.no_grad():expected=deployment(torch.from_numpy(features),torch.from_numpy(usable)).numpy()
    actual=session.run(None,{'features':features,'usable':usable})[0]
    np.testing.assert_allclose(actual,expected,rtol=1e-4,atol=2e-5)
    errors.append(float(np.max(np.abs(actual-expected))))
metadata=dict(schema='motionlab-exercise-onnx-v1',architecture='plain_tcn',parameters=15894,
    classes=module.CLASSES,source_feature_names=module.EXPECTED_SOURCE_NAMES,
    frames_per_window=45,target_hz=15,prediction_stride_observations=8,
    minimum_window_coverage=.8,max_render_edge=1280,
    checkpoint_sha256=module.FROZEN_CHECKPOINTS['plain_tcn'],
    onnx_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    selected_epoch=state['epoch'],training_protocol=protocol,
    export_verification=dict(cases=20,max_absolute_logit_error=max(errors)),
    limits='Closed six-class classifier. Scores are uncalibrated. No clinical/form-safety claim. '
        'GUI streaming smoothing and repetition routing are new prototype behavior, not the held-out window benchmark.')
(out/'exercise_plain_tcn.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
print('Exported:',path,'bytes:',path.stat().st_size,'max logit error:',max(errors))
