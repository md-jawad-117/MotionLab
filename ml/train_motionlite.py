"""GPU training on cached MediaPipe windows. No pose inference or test evaluation."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import os
import random
import shutil
import time
import zipfile

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

VERSION = 'motionlite-v1.0.0'
SOURCE_DIGEST = 'f1489b3dfd76726248cde4d6c353272e984fc82e34d7ad7d4983ada722eb8bdc'
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


def save_torch(path, value):
    temporary = path.with_name(path.name+'.tmp')
    torch.save(value, temporary); temporary.replace(path)


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


def load_train_validation(root):
    config = json.loads((root/'config.json').read_text())
    if config['classes']!=CLASSES or config['target_hz']!=15 or round(config['window_s']*15)!=45:
        raise ValueError('Unsupported feature/window configuration')
    with (root/'windows.csv').open(newline='',encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    counts = validate_rows(rows)
    decisions = json.loads((root/'review_decisions.json').read_text())
    approved = {r['id']:r for r in decisions['intervals'] if r.get('approved')}
    selected = [r for r in rows if r['split'] in ('train','validation')]
    # Only training/validation caches are opened. Test arrays remain unopened.
    by_session = {}
    for row in selected:
        by_session.setdefault(row['session'],[]).append(row)
    sizes = {role:sum(counts[role].values()) for role in ('train','validation')}
    arrays = {role:dict(X=np.empty((n,45,92),np.float32),y=np.empty(n,np.int64),
        usable=np.empty((n,45),bool)) for role,n in sizes.items()}
    cursor = {role:0 for role in arrays}; fingerprints = {}
    for number,(session,group) in enumerate(sorted(by_session.items()),1):
        print(f'Loading train/validation cache {number}/{len(by_session)}: {session}',flush=True)
        names = {Path(r['cache_path']).name for r in group}
        if len(names)!=1:
            raise ValueError(f'Ambiguous cache names for {session}')
        cache = root/'features'/next(iter(names))
        meta = json.loads(cache.with_suffix('.json').read_text())
        digest = sha(cache)
        if digest!=meta['cache_sha256']:
            raise ValueError(f'Corrupt cache: {session}')
        fp = meta['fingerprint']
        if fp['config']['target_hz']!=15 or fp['feature_names']!=EXPECTED_SOURCE_NAMES:
            raise ValueError('Unsupported source features')
        fingerprints[session] = digest
        with np.load(cache,allow_pickle=False) as source:
            reduced = source['features'][:,SOURCE_COLUMNS]
            frames,times,usable = source['frames'],source['times'],source['usable']
            if not np.isfinite(reduced).all():
                raise ValueError('Nonfinite source features')
            for row in group:
                role = row['split']; start,end = int(row['start_index']),int(row['end_index'])
                if end>len(reduced) or int(frames[start])!=int(row['start_frame']) or int(frames[end-1])!=int(row['last_sample_frame']):
                    raise ValueError(f'Window offset/frame mismatch: {session}')
                if np.any(np.diff(times[start:end])>1.6/15) or float(usable[start:end].mean())<config['minimum_window_coverage']:
                    raise ValueError('Window continuity/coverage failed')
                if int(row['class_id'])==5:
                    decision=approved.get(row['review_id'],{})
                    if (decision.get('session')!=session or decision.get('split')!=role or
                            decision.get('cache_sha256')!=digest or
                            int(frames[start])<decision.get('start_frame',float('inf')) or
                            int(frames[end-1])>=decision.get('end_frame_exclusive',-1)):
                        raise ValueError(f'Background interval approval does not match: {session}')
                index = cursor[role]; cursor[role]+=1
                arrays[role]['X'][index] = reduced[start:end]
                arrays[role]['y'][index] = int(row['class_id'])
                arrays[role]['usable'][index] = usable[start:end]
    identity = dict(index_sha256=sha(root/'windows.csv'),config_sha256=sha(root/'config.json'),
        loaded_cache_sha256=fingerprints,classes=CLASSES,body_joint_ids=BODY,
        source_columns=SOURCE_COLUMNS,source_dimension=239,model_input_dimension=92,
        frames_per_window=45,target_hz=15,test_feature_arrays_loaded=False)
    return arrays,identity,{role:{CLASSES[i]:counts[role][i] for i in range(6)} for role in counts}


def fit_normalization(training):
    # Statistics use accepted TRAIN windows and valid observations only.
    total=np.zeros(48,np.float64); squared=total.copy(); count=total.copy()
    for first in range(0,len(training),256):
        block=training[first:first+256]
        mask=np.concatenate((np.repeat(block[:,:,60:72],2,axis=2),
            np.repeat(block[:,:,72:84],2,axis=2)),axis=2)>0.5
        values=block[:,:,:48].astype(np.float64)
        total+=(values*mask).sum(axis=(0,1)); squared+=(values**2*mask).sum(axis=(0,1))
        count+=mask.sum(axis=(0,1))
    mean=total/np.maximum(count,1)
    variance=np.maximum(0,squared/np.maximum(count,1)-mean**2)
    std=np.sqrt(variance); std[std<1e-4]=1
    return mean.astype(np.float32),std.astype(np.float32)


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


def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)


def rng_snapshot(generator):
    numpy_state=np.random.get_state()
    return dict(python=random.getstate(),numpy=[numpy_state[0],numpy_state[1].tolist(),
        numpy_state[2],numpy_state[3],numpy_state[4]],torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        loader=generator.get_state())


def restore_rng(state,generator):
    random.setstate(state['python'])
    n=state['numpy'];np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(state['torch']);generator.set_state(state['loader'])
    if torch.cuda.is_available() and state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


@torch.no_grad()
def evaluate(model,loader,device):
    model.eval();actual=[];predicted=[];loss_sum=0;total=0
    for x,y,usable in loader:
        x,y,usable=x.to(device,non_blocking=True),y.to(device,non_blocking=True),usable.to(device,non_blocking=True)
        logits=model(x,usable)
        loss_sum+=float(F.cross_entropy(logits,y,reduction='sum').item());total+=len(y)
        actual.extend(y.cpu().tolist());predicted.extend(logits.argmax(1).cpu().tolist())
    metrics=classification_metrics(np.array(actual),np.array(predicted));metrics['loss']=loss_sum/max(total,1)
    return metrics


def train_model(name,model,arrays,identity,normalization,settings,output,device):
    seed_everything(settings['seed'])
    model=model.to(device);parameters=sum(p.numel() for p in model.parameters())
    if name=='motionlite' and parameters>=100000:
        raise ValueError('MotionLite exceeds the intended parameter budget')
    folder=output/name;folder.mkdir(parents=True,exist_ok=True)
    protocol=dict(version=VERSION,source_digest=SOURCE_DIGEST,dataset=identity,settings=settings,
        architecture=name,parameters=parameters,normalization={k:v.tolist() for k,v in normalization.items()})
    protocol_path=folder/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text())!=protocol:
        raise ValueError(f'Run settings/data changed: {folder}. Use a new training output folder.')
    save_json(protocol_path,protocol)
    train=arrays['train'];valid=arrays['validation']
    generator=torch.Generator().manual_seed(settings['seed'])
    def dataset(part):
        return TensorDataset(torch.from_numpy(part['X']),torch.from_numpy(part['y']),torch.from_numpy(part['usable']))
    loader=DataLoader(dataset(train),batch_size=settings['batch_size'],shuffle=True,
        generator=generator,num_workers=0,pin_memory=device.type=='cuda')
    validation=DataLoader(dataset(valid),batch_size=256,shuffle=False,num_workers=0,pin_memory=device.type=='cuda')
    count=np.bincount(train['y'],minlength=6)
    weights=1/np.sqrt(count.astype(np.float32));weights/=weights.mean()
    criterion=nn.CrossEntropyLoss(weight=torch.tensor(weights,device=device))
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=4)
    last=folder/'last.pt';best=folder/'best.pt';history=[];best_score=-1.;stale=0;first_epoch=1;finished=False
    if last.exists():
        state=torch.load(last,map_location='cpu',weights_only=True)
        if state['protocol']!=protocol:raise ValueError('Checkpoint protocol differs')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler']);history=state['history']
        best_score=state['best_score'];stale=state['stale'];first_epoch=state['epoch']+1
        finished=state.get('finished',False);restore_rng(state['rng'],generator)
        print(f'{name}: resuming after epoch {state["epoch"]}',flush=True)
    print(f'\n{name}: {parameters:,} parameters on {device}; train={len(train["y"]):,}, validation={len(valid["y"]):,}',flush=True)
    if not finished:
        for epoch in range(first_epoch,settings['epochs']+1):
            start=time.monotonic();model.train();loss_sum=0.;samples=0
            for x,y,usable in loader:
                x,y,usable=x.to(device,non_blocking=True),y.to(device,non_blocking=True),usable.to(device,non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits=model(x,usable);loss=criterion(logits,y)
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
                loss_sum+=loss.item()*len(y);samples+=len(y)
            metrics=evaluate(model,validation,device);score=metrics['macro_f1']
            scheduler.step(score)
            row=dict(epoch=epoch,train_weighted_loss=loss_sum/samples,
                validation_loss=metrics['loss'],validation_accuracy=metrics['accuracy'],
                validation_macro_f1=score,learning_rate=optimizer.param_groups[0]['lr'],
                epoch_seconds=time.monotonic()-start)
            history.append(row)
            if score>best_score+1e-6:
                best_score=score;stale=0
                save_torch(best,dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                    protocol=protocol,epoch=epoch,validation=metrics))
            else:stale+=1
            finished=stale>=settings['patience'] or epoch==settings['epochs']
            save_torch(last,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),protocol=protocol,epoch=epoch,history=history,
                best_score=best_score,stale=stale,finished=finished,rng=rng_snapshot(generator)))
            with (folder/'history.csv').open('w',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
            print(f'{name} epoch {epoch:02d}: val F1={score:.4f}, accuracy={metrics["accuracy"]:.4f}, '
                  f'{row["epoch_seconds"]:.1f}s',flush=True)
            if finished:break
    if not best.exists():raise ValueError('No best-model checkpoint was saved')
    state=torch.load(best,map_location='cpu',weights_only=True)
    model.load_state_dict(state['model']);metrics=evaluate(model,validation,device)
    result=dict(architecture=name,parameters=parameters,best_epoch=state['epoch'],validation=metrics,
        class_weights=weights.tolist(),best_checkpoint=str(best),test_evaluation='not_run',
        limitations='One seed; overlapping windows; provisional background labels; session protocol and frame-origin caveats retained.')
    save_json(folder/'metrics.json',result)
    return result,history


def plot_results(results,histories,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(1,2,figsize=(11,4))
    for name,history in histories.items():
        axes[0].plot([r['epoch'] for r in history],[r['validation_macro_f1'] for r in history],label=name)
        axes[1].plot([r['epoch'] for r in history],[r['validation_loss'] for r in history],label=name)
    axes[0].set(ylabel='Validation macro-F1',xlabel='Epoch',ylim=(0,1))
    axes[1].set(ylabel='Validation cross-entropy',xlabel='Epoch')
    for axis in axes:axis.grid(alpha=.2);axis.legend()
    figure.tight_layout();figure.savefig(output/'training_curves.png',dpi=150);plt.close(figure)
    matrix=np.array(results['motionlite']['validation']['confusion_matrix'])
    figure,axis=plt.subplots(figsize=(8,6));im=axis.imshow(matrix,cmap='Blues')
    axis.set(xticks=range(6),yticks=range(6),xticklabels=CLASSES,yticklabels=CLASSES,
        xlabel='Predicted',ylabel='Reference label',title='MotionLite validation counts')
    plt.setp(axis.get_xticklabels(),rotation=35,ha='right')
    for i in range(6):
        for j in range(6):axis.text(j,i,str(matrix[i,j]),ha='center',va='center')
    figure.colorbar(im,ax=axis);figure.tight_layout();figure.savefig(output/'validation_confusion.png',dpi=150);plt.close(figure)


def main():
    allow_cpu=os.environ.get('MOTIONLITE_ALLOW_CPU')=='1'  # local software tests only
    if not torch.cuda.is_available() and not allow_cpu:
        raise RuntimeError('GPU is not visible to PyTorch. Enable Kaggle GPU before running this training cell.')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cpu':torch.set_num_threads(2)
    print('TRAINING runtime:',torch.__version__, '| device:',device,
        '| GPU:',torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU software test',flush=True)
    output=Path(os.environ.get('MOTIONLITE_OUTPUT','/kaggle/working/motionlite_training'))
    output.mkdir(parents=True,exist_ok=True)
    root=find_dataset();arrays,identity,counts=load_train_validation(root)
    mean,std=fit_normalization(arrays['train']['X'])
    for part in arrays.values():normalize_in_place(part['X'],mean,std)
    settings=dict(seed=42,epochs=int(os.environ.get('MOTIONLITE_EPOCHS','60')),
        patience=10,batch_size=128,learning_rate=.002,weight_decay=.001,
        selection='validation_macro_f1',class_weighting='inverse_sqrt_training_counts',
        device_type=device.type,torch_version=str(torch.__version__),
        accelerators=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if device.type=='cuda' else [])
    save_json(output/'dataset_manifest.json',dict(identity=identity,window_counts=counts,
        data_root=str(root),normalization_fit='accepted training windows only',
        test_feature_arrays_loaded=False))
    results={};histories={}
    for name,architecture in [('plain_tcn',PlainTCN),('motionlite',MotionLite)]:
        seed_everything(settings['seed'])
        result,history=train_model(name,architecture(),arrays,identity,dict(mean=mean,std=std),
            settings,output,device)
        results[name]=result;histories[name]=history
        save_json(output/'results.json',dict(results=results,test_evaluation='not_run',
            training_complete=len(results)==2))
    plot_results(results,histories,output)
    # Include the readable architecture/trainer source in the downloadable run.
    try:
        source=Path(__file__)
        if source.is_file() and source.name!='ipykernel_launcher.py':
            shutil.copy2(source,output/'train_motionlite.py')
    except NameError:
        pass
    archive=output.parent/'MotionLite_Training_Results.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(output.rglob('*')):
            if path.is_file() and not path.name.endswith('.tmp'):
                bundle.write(path,path.relative_to(output.parent))
    print('\nTRAINING COMPLETE. Validation results only:',flush=True)
    print(json.dumps({name:dict(parameters=r['parameters'],best_epoch=r['best_epoch'],
        validation_accuracy=r['validation']['accuracy'],validation_macro_f1=r['validation']['macro_f1'])
        for name,r in results.items()},indent=2),flush=True)
    print('Test splits were not evaluated. Save/download:',archive,flush=True)
    try:
        from IPython.display import display,FileLink
        display(FileLink(str(archive)))
    except ImportError:
        pass


if __name__=='__main__':
    main()
