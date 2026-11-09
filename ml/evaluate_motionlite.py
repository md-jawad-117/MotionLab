"""Fixed held-out evaluation. No optimizer, training loop or normalization fitting."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import os
import shutil
import zipfile
os.environ['MPLBACKEND']='Agg'
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


CLASSES = ['squats','pushups','lunges','bicep_curls','other_exercise','non_activity']

BODY = [11,12,13,14,15,16,23,24,25,26,27,28]

SOURCE_COLUMNS = ([2*j+a for j in BODY for a in (0,1)] +
    [66+2*j+a for j in BODY for a in (0,1)] +
    [132+j for j in BODY] + [165+j for j in BODY] + [198+j for j in BODY] + list(range(231,239)))

EXPECTED_SOURCE_NAMES = ([f'position_{j}_{a}' for j in range(33) for a in ('x','y')] +
    [f'velocity_{j}_{a}' for j in range(33) for a in ('x','y')] +
    [f'quality_{j}' for j in range(33)] + [f'valid_{j}' for j in range(33)] +
    [f'velocity_valid_{j}' for j in range(33)] +
    ['left_knee','right_knee','left_elbow','right_elbow'] +
    ['left_knee_valid','right_knee_valid','left_elbow_valid','right_elbow_valid'])

ROLES = {
    'train':['w01','w02','w03','w04','w06','w07','w08','w16','w17','w18'],
    'validation':['w14','w15','w19'],
    'seen_session_test':['w09','w10','w11'],
    'unseen_subject_test':['w00','w05','w12','w13','w20'],
}

ROLE_OF = {session:role for role, sessions in ROLES.items() for session in sessions}

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()

def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)

def unpack_dataset(archive, destination):
    with zipfile.ZipFile(archive) as bundle:
        plans = []
        for member in bundle.infolist():
            name = member.filename
            target = (destination/name).resolve()
            if ('\\' in name or ':' in name or not target.is_relative_to(destination.resolve()) or
                    ((member.external_attr >> 16) & 0o170000) == 0o120000):
                raise ValueError(f'Unsafe checkpoint entry: {name}')
            if not member.is_dir() and Path(name).suffix in {'.npz','.json','.csv'}:
                plans.append((member,target))
        for member, target in plans:
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                continue
            temporary = target.with_name(target.name+'.restore')
            with bundle.open(member) as source, temporary.open('wb') as output:
                shutil.copyfileobj(source,output)
            temporary.replace(target)

def find_dataset():
    override = os.environ.get('MOTIONLITE_DATA')
    preferred = Path(override or '/kaggle/working/motionlab_pipeline')
    if (preferred/'windows.csv').is_file():
        return preferred.resolve()
    if override:
        raise ValueError(f'No windows.csv at {preferred}')
    inputs = Path('/kaggle/input')
    ready = [p.parent for p in inputs.rglob('windows.csv')
        if (p.parent/'features').is_dir() and (p.parent/'report.json').is_file()]
    ready = [p for p in ready if json.loads((p/'report.json').read_text()).get('sessions')==21]
    if len(ready)==1:
        return ready[0].resolve()
    if len(ready)>1:
        raise ValueError('Several datasets found; set MOTIONLITE_DATA to the intended folder.')
    archives = []
    for path in inputs.rglob('*.zip'):
        if 'checkpoint' not in path.name.lower():
            continue
        with zipfile.ZipFile(path) as bundle:
            if any(n.endswith('motionlab_pipeline/windows.csv') for n in bundle.namelist()):
                archives.append(path)
    local = Path('/kaggle/working/motionlab_pipeline_checkpoint.zip')
    if local.is_file():
        archives = [local]
    if len(archives)!=1:
        raise ValueError('Attach motionlab_pipeline_checkpoint.zip as a Kaggle input. '
                         f'Found {len(archives)} suitable archives.')
    restored = Path('/kaggle/working/motionlite_restored_dataset')
    print('Restoring feature dataset; no video processing...',flush=True)
    unpack_dataset(archives[0],restored)
    matches = list(restored.rglob('motionlab_pipeline/windows.csv'))
    if len(matches)!=1:
        raise ValueError('Could not identify the restored dataset.')
    return matches[0].parent.resolve()

def validate_rows(rows):
    seen = set()
    counts = {role:Counter() for role in ROLES}
    for row in rows:
        session, role = row['session'],row['split']
        if ROLE_OF.get(session)!=role:
            raise ValueError(f'Unexpected split assignment: {session}, {role}')
        class_id = int(row['class_id'])
        if class_id not in range(6) or row['class_name']!=CLASSES[class_id]:
            raise ValueError('Class schema mismatch')
        start,end = int(row['start_index']),int(row['end_index'])
        if start<0 or end-start!=45:
            raise ValueError('Require 45-observation windows')
        if class_id==5 and not row.get('review_id'):
            raise ValueError('Background window lacks an interval review ID')
        key = session,row['cache_id'],start,end
        if key in seen:
            raise ValueError(f'Duplicate window: {key}')
        seen.add(key); counts[role][class_id]+=1
    for role in ('train','validation'):
        if any(counts[role][i]==0 for i in range(6)):
            raise ValueError(f'{role} does not contain all six classes')
    return counts

def normalize_in_place(values,mean,std):
    for first in range(0,len(values),256):
        block=values[first:first+256]
        mask=np.concatenate((np.repeat(block[:,:,60:72],2,axis=2),
            np.repeat(block[:,:,72:84],2,axis=2)),axis=2)>0.5
        block[:,:,:48]=np.where(mask,np.clip((block[:,:,:48]-mean)/std,-8,8),0)

class CausalBlock(nn.Module):
    def __init__(self,channels,dilation,dropout=.15):
        super().__init__();self.padding=4*dilation
        self.depthwise=nn.Conv1d(channels,channels,5,dilation=dilation,groups=channels)
        self.pointwise=nn.Conv1d(channels,channels,1)
        self.norm=nn.LayerNorm(channels);self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        y=self.pointwise(F.silu(self.depthwise(F.pad(x,(self.padding,0)))))
        y=self.norm(y.transpose(1,2)).transpose(1,2)
        return x+self.dropout(F.silu(y))

def angle_stream(x):
    angles=x[:,:,84:88];valid=x[:,:,88:92]
    delta=torch.zeros_like(angles)
    delta[:,1:]=(angles[:,1:]-angles[:,:-1])*15*valid[:,1:]*valid[:,:-1]
    return torch.cat((angles,delta.clamp(-20,20),valid),dim=2)

def temporal_pool(x,usable):
    weight=usable.to(x.dtype).unsqueeze(1)
    return (x*weight).sum(dim=2)/weight.sum(dim=2).clamp_min(1)

class MotionLite(nn.Module):
    """Two compact causal streams with learned tracking-quality fusion gates."""
    def __init__(self):
        super().__init__()
        self.pose=nn.Sequential(nn.Conv1d(84,32,1),nn.SiLU(),
            *[CausalBlock(32,d) for d in (1,2,4,8)])
        self.angles=nn.Sequential(nn.Conv1d(12,16,1),nn.SiLU(),
            *[CausalBlock(16,d) for d in (1,2,4,8)])
        self.gate=nn.Sequential(nn.Linear(4,16),nn.SiLU(),nn.Linear(16,2),nn.Sigmoid())
        self.head=nn.Sequential(nn.Linear(48,32),nn.SiLU(),nn.Dropout(.15),nn.Linear(32,6))
    def forward(self,x,usable):
        quality=torch.stack((x[:,:,48:60].mean(2),x[:,:,60:72].mean(2),
            x[:,:,72:84].mean(2),x[:,:,88:92].mean(2)),dim=2)
        gates=self.gate(quality).transpose(1,2)
        pose=self.pose(x[:,:,:84].transpose(1,2))*gates[:,0:1]
        angles=self.angles(angle_stream(x).transpose(1,2))*gates[:,1:2]
        return self.head(temporal_pool(torch.cat((pose,angles),dim=1),usable))

class PlainTCN(nn.Module):
    """Single-stream comparison with the same pose/angle/delta information."""
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv1d(96,48,1),nn.SiLU(),
            *[CausalBlock(48,d) for d in (1,2,4,8)])
        self.head=nn.Linear(48,6)
    def forward(self,x,usable):
        values=torch.cat((x[:,:,:84],angle_stream(x)),dim=2).transpose(1,2)
        return self.head(temporal_pool(self.encoder(values),usable))

def classification_metrics(actual,predicted):
    matrix=np.zeros((6,6),np.int64)
    np.add.at(matrix,(actual,predicted),1)
    tp=matrix.diagonal();support=matrix.sum(1);pred_count=matrix.sum(0)
    precision=tp/np.maximum(pred_count,1);recall=tp/np.maximum(support,1)
    f1=2*precision*recall/np.maximum(precision+recall,1e-12)
    return dict(accuracy=float(tp.sum()/max(1,matrix.sum())),macro_f1=float(f1.mean()),
        confusion_matrix=matrix.tolist(),classes=CLASSES,per_class={name:dict(
        precision=float(precision[i]),recall=float(recall[i]),f1=float(f1[i]),support=int(support[i]))
        for i,name in enumerate(CLASSES)})

# Appended to exact, tested model/preprocessing definitions by build_evaluation.py.
FROZEN_CHECKPOINTS = {
    'plain_tcn':'4a59186a43a798a45821a41d0e465121fd0e01b4848d24dd7643acd2e89e51d2',
    'motionlite':'ced77dbe8e92f417d314c2bc724f49ed65806d5975d6c9cdbd0bcfef8b396646',
}
FROZEN_TRAINER_DIGEST = 'f1489b3dfd76726248cde4d6c353272e984fc82e34d7ad7d4983ada722eb8bdc'
TEST_ROLES = ('seen_session_test','unseen_subject_test')


def unpack_models(archive,destination):
    with zipfile.ZipFile(archive) as bundle:
        plans=[]
        for member in bundle.infolist():
            target=(destination/member.filename).resolve()
            if ('\\' in member.filename or ':' in member.filename or
                    not target.is_relative_to(destination.resolve()) or
                    ((member.external_attr >> 16) & 0o170000)==0o120000):
                raise ValueError('Unsafe model archive entry')
            if not member.is_dir() and Path(member.filename).suffix in {'.pt','.json','.csv'}:
                plans.append((member,target))
        for member,target in plans:
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():
                temporary=target.with_name(target.name+'.restore')
                with bundle.open(member) as source,temporary.open('wb') as output:
                    shutil.copyfileobj(source,output)
                temporary.replace(target)


def find_models():
    override=os.environ.get('MOTIONLITE_MODELS')
    root=Path(override or '/kaggle/working/motionlite_training')
    if all((root/name/'best.pt').is_file() for name in FROZEN_CHECKPOINTS):
        return root.resolve()
    if override:raise ValueError(f'Both best.pt files are required under {root}')
    inputs=Path('/kaggle/input')
    candidates=sorted(set(p.parent.parent for p in inputs.rglob('motionlite/best.pt')
        if (p.parent.parent/'plain_tcn/best.pt').is_file()))
    if len(candidates)==1:return candidates[0].resolve()
    if len(candidates)>1:raise ValueError('Multiple model folders found; set MOTIONLITE_MODELS.')
    archives=[p for p in inputs.rglob('*.zip') if p.name.lower()=='motionlite_training_results.zip']
    local=Path('/kaggle/working/MotionLite_Training_Results.zip')
    if local.is_file():archives=[local]
    if len(archives)!=1:
        raise ValueError('Attach MotionLite_Training_Results.zip as a Kaggle input.')
    restored=Path('/kaggle/working/motionlite_restored_models')
    unpack_models(archives[0],restored)
    candidates=[p.parent.parent for p in restored.rglob('motionlite/best.pt')
        if (p.parent.parent/'plain_tcn/best.pt').is_file()]
    if len(candidates)!=1:raise ValueError('Could not identify restored best checkpoints.')
    return candidates[0].resolve()


def load_frozen_models(folder,device):
    models={};protocols={};reference=None
    for name,architecture in [('plain_tcn',PlainTCN),('motionlite',MotionLite)]:
        path=folder/name/'best.pt'
        if sha(path)!=FROZEN_CHECKPOINTS[name]:
            raise ValueError(f'{name} best.pt differs from the frozen uploaded training run. '
                'Use the original MotionLite_Training_Results.zip; no evaluation started.')
        state=torch.load(path,map_location='cpu',weights_only=True)
        protocol=state['protocol']
        if (protocol['architecture']!=name or protocol['source_digest']!=FROZEN_TRAINER_DIGEST or
                protocol['dataset']['classes']!=CLASSES or
                protocol['dataset']['source_columns']!=SOURCE_COLUMNS or
                protocol['settings']['selection']!='validation_macro_f1'):
            raise ValueError('Unsupported frozen model protocol')
        if reference is None:reference=protocol['dataset']
        if reference!=protocol['dataset']:raise ValueError('Models used different datasets')
        mean=np.array(protocol['normalization']['mean'],np.float32)
        std=np.array(protocol['normalization']['std'],np.float32)
        if mean.shape!=(48,) or std.shape!=(48,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std<=0):
            raise ValueError('Invalid saved normalization')
        model=architecture().to(device);model.load_state_dict(state['model'],strict=True)
        model.eval();model.requires_grad_(False)
        if sum(p.numel() for p in model.parameters())!=protocol['parameters']:
            raise ValueError('Parameter count differs from checkpoint protocol')
        models[name]=model;protocols[name]=dict(protocol=protocol,selected_epoch=state['epoch'])
        print(f'Frozen {name}: validation-selected epoch {state["epoch"]}, '
            f'{protocol["parameters"]:,} parameters',flush=True)
    return models,protocols


def load_test_windows(root,identity):
    if sha(root/'windows.csv')!=identity['index_sha256'] or sha(root/'config.json')!=identity['config_sha256']:
        raise ValueError('Dataset index/config differs from the frozen training dataset. '
            'Restore the original preparation checkpoint.')
    config=json.loads((root/'config.json').read_text())
    if config['classes']!=CLASSES or config['target_hz']!=15:
        raise ValueError('Unsupported dataset schema')
    with (root/'windows.csv').open(newline='',encoding='utf-8') as handle:
        rows=list(csv.DictReader(handle))
    counts=validate_rows(rows)
    selected=[r for r in rows if r['split'] in TEST_ROLES]
    if {r['session'] for r in selected} != {s for role in TEST_ROLES for s in ROLES[role]}:
        raise ValueError('Require all eight expected held-out workouts')
    if any(counts[role][i]==0 for role in TEST_ROLES for i in range(6)):
        raise ValueError('A held-out split is missing a class')
    decisions=json.loads((root/'review_decisions.json').read_text())
    approved={r['id']:r for r in decisions['intervals'] if r.get('approved')}
    arrays={role:dict(X=np.empty((sum(counts[role].values()),45,92),np.float32),
        y=np.empty(sum(counts[role].values()),np.int64),usable=np.empty((sum(counts[role].values()),45),bool),
        rows=[]) for role in TEST_ROLES}
    groups={}
    for row in selected:groups.setdefault(row['session'],[]).append(row)
    cursor={role:0 for role in TEST_ROLES};hashes={}
    for number,(session,group) in enumerate(sorted(groups.items()),1):
        print(f'Loading held-out cache {number}/8: {session}',flush=True)
        names={Path(r['cache_path']).name for r in group}
        if len(names)!=1:raise ValueError(f'Ambiguous cache for {session}')
        cache=root/'features'/next(iter(names))
        meta=json.loads(cache.with_suffix('.json').read_text());digest=sha(cache)
        if digest!=meta['cache_sha256']:raise ValueError(f'Corrupt test cache: {session}')
        fp=meta['fingerprint']
        if fp['feature_names']!=EXPECTED_SOURCE_NAMES or fp['config']['target_hz']!=15:
            raise ValueError('Unsupported feature layout')
        hashes[session]=digest
        with np.load(cache,allow_pickle=False) as source:
            reduced=source['features'][:,SOURCE_COLUMNS]
            frames,times,usable=source['frames'],source['times'],source['usable']
            if not np.isfinite(reduced).all():raise ValueError('Nonfinite test features')
            for row in group:
                role=row['split'];a,b=int(row['start_index']),int(row['end_index'])
                if b>len(frames) or int(frames[a])!=int(row['start_frame']) or int(frames[b-1])!=int(row['last_sample_frame']):
                    raise ValueError('Test window offsets do not match the cache')
                if np.any(np.diff(times[a:b])>1.6/15) or usable[a:b].mean()<config['minimum_window_coverage']:
                    raise ValueError('Test continuity/coverage failed')
                if int(row['class_id'])==5:
                    decision=approved.get(row['review_id'],{})
                    if (decision.get('session')!=session or decision.get('split')!=role or
                            decision.get('cache_sha256')!=digest or
                            int(frames[a])<decision.get('start_frame',float('inf')) or
                            int(frames[b-1])>=decision.get('end_frame_exclusive',-1)):
                        raise ValueError('Test background approval does not match the exact interval')
                index=cursor[role];cursor[role]+=1
                arrays[role]['X'][index]=reduced[a:b];arrays[role]['y'][index]=int(row['class_id'])
                arrays[role]['usable'][index]=usable[a:b];arrays[role]['rows'].append(row)
    return arrays,dict(loaded_test_cache_sha256=hashes,training_feature_arrays_loaded=False,
        index_sha256=identity['index_sha256'],config_sha256=identity['config_sha256'])


@torch.no_grad()
def predict(model,part,normalization,device):
    x=part['X'].copy()
    normalize_in_place(x,np.array(normalization['mean'],np.float32),np.array(normalization['std'],np.float32))
    loader=DataLoader(TensorDataset(torch.from_numpy(x),torch.from_numpy(part['usable'])),
        batch_size=256,shuffle=False,num_workers=0,pin_memory=device.type=='cuda')
    model.eval();probabilities=[]
    for values,usable in loader:
        logits=model(values.to(device,non_blocking=True),usable.to(device,non_blocking=True))
        probabilities.append(torch.softmax(logits,dim=1).cpu().numpy())
    result=np.concatenate(probabilities)
    if not np.isfinite(result).all():raise ValueError('Nonfinite predicted probabilities')
    return result


def export_predictions(path,rows,actual,probabilities):
    predicted=probabilities.argmax(1)
    fields=['session','split','cache_id','start_frame','last_sample_frame','start_s','last_sample_s',
        'original_activity','reference_class','predicted_class','max_softmax_uncalibrated']+['p_'+name for name in CLASSES]
    with path.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for i,row in enumerate(rows):
            record={k:row.get(k,'') for k in fields[:8]}
            record.update(reference_class=CLASSES[int(actual[i])],predicted_class=CLASSES[int(predicted[i])],
                max_softmax_uncalibrated=float(probabilities[i].max()))
            record.update({'p_'+name:float(probabilities[i,j]) for j,name in enumerate(CLASSES)})
            writer.writerow(record)


def plot_test_results(results,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name,by_role in results.items():
        for role,metrics in by_role.items():
            matrix=np.array(metrics['confusion_matrix'])
            figure,axis=plt.subplots(figsize=(8,6));image=axis.imshow(matrix,cmap='Blues')
            axis.set(xticks=range(6),yticks=range(6),xticklabels=CLASSES,yticklabels=CLASSES,
                xlabel='Predicted class',ylabel='CSV/review reference class',title=f'{name}: {role}')
            plt.setp(axis.get_xticklabels(),rotation=35,ha='right')
            for i in range(6):
                for j in range(6):axis.text(j,i,str(matrix[i,j]),ha='center',va='center')
            figure.colorbar(image,ax=axis);figure.tight_layout()
            figure.savefig(output/f'{name}_{role}_confusion.png',dpi=150);plt.close(figure)


def main():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cpu':torch.set_num_threads(2)
    print('FINAL EVALUATION ONLY — no retraining or normalization fitting.',flush=True)
    print('Device:',device, '| Torch:',torch.__version__,flush=True)
    output=Path(os.environ.get('MOTIONLITE_TEST_OUTPUT','/kaggle/working/motionlite_test_evaluation'))
    output.mkdir(parents=True,exist_ok=True)
    folder=find_models();models,protocols=load_frozen_models(folder,device)
    identity=protocols['motionlite']['protocol']['dataset']
    plan=dict(protocol_version='motionlite-fixed-test-v1',primary_model='motionlite',
        comparison_model='plain_tcn',checkpoint_sha256=FROZEN_CHECKPOINTS,
        selected_epochs={name:state['selected_epoch'] for name,state in protocols.items()},
        selection_rule='best validation macro-F1 from the completed training run',
        test_sessions={role:ROLES[role] for role in TEST_ROLES},
        prediction_rule='argmax of six output logits; no test-tuned rejection threshold',
        preprocessing='saved training mean/std; never refit on test data',
        dataset_index_sha256=identity['index_sha256'],dataset_config_sha256=identity['config_sha256'])
    plan_path=output/'frozen_evaluation_plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text())!=plan:
        raise ValueError('This output folder already contains a different frozen evaluation plan')
    save_json(plan_path,plan)  # Committed before any test predictions are computed.
    root=find_dataset();arrays,data_identity=load_test_windows(root,identity)
    save_json(output/'data_manifest.json',data_identity)
    results={};summary=[];per_class=[];per_session=[]
    for name,model in models.items():
        results[name]={};saved=protocols[name]['protocol']
        for role,part in arrays.items():
            print(f'Evaluating {name} on {role}: {len(part["y"]):,} windows...',flush=True)
            probabilities=predict(model,part,saved['normalization'],device)
            predicted=probabilities.argmax(1);metrics=classification_metrics(part['y'],predicted)
            metrics.update(windows=len(predicted),sessions=sorted({r['session'] for r in part['rows']}))
            results[name][role]=metrics
            summary.append(dict(model=name,split=role,windows=len(predicted),
                accuracy=metrics['accuracy'],macro_f1=metrics['macro_f1']))
            for class_name,values in metrics['per_class'].items():
                per_class.append(dict(model=name,split=role,class_name=class_name,**values))
            for session in metrics['sessions']:
                indices=np.array([r['session']==session for r in part['rows']])
                session_metrics=classification_metrics(part['y'][indices],predicted[indices])
                per_session.append(dict(model=name,split=role,session=session,windows=int(indices.sum()),
                    accuracy=session_metrics['accuracy'],macro_f1=session_metrics['macro_f1']))
            export_predictions(output/f'{name}_{role}_predictions.csv',part['rows'],part['y'],probabilities)
    for filename,rows in [('summary.csv',summary),('per_class.csv',per_class),('per_session.csv',per_session)]:
        with (output/filename).open('w',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report=dict(frozen_plan=plan,results=results,parameters={n:p['protocol']['parameters'] for n,p in protocols.items()},
        runtime=dict(torch=str(torch.__version__),device=str(device),
            gpu=torch.cuda.get_device_name(0) if device.type=='cuda' else None),
        training_performed=False,normalization_refitted=False,test_threshold_tuning=False,
        limits='One seed and correlated overlapping windows. Background labels are sparse spot-checks; '
            'frame-origin and viewpoint limitations remain. Split names follow the existing MM-Fit '
            'protocol; workout count is not subject count. These results are window classification, '
            'not repetition-count accuracy, calibrated uncertainty or proof of population performance.')
    save_json(output/'test_results.json',report)
    plot_test_results(results,output)
    archive=output.parent/'MotionLite_Test_Evaluation_Results.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(output.rglob('*')):
            if path.is_file() and not path.name.endswith('.tmp'):
                bundle.write(path,path.relative_to(output.parent))
    print('\nFINAL TEST EVALUATION COMPLETE',flush=True)
    print(json.dumps(summary,indent=2),flush=True)
    print('Save/download:',archive,flush=True)
    try:
        from IPython.display import display,FileLink
        display(FileLink(str(archive)))
    except ImportError:pass


if __name__=='__main__':
    main()
