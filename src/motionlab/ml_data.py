"""Shared causal MediaPipe feature schema and label-safe window construction."""
import csv,hashlib,json,math
from pathlib import Path
import numpy as np
from .geometry import angle

CLASSES=('squats','pushups','lunges','bicep_curls','other_exercise','non_activity')
ACTIVITIES=set(CLASSES[:4])|{'dumbbell_rows','dumbbell_shoulder_press','situps','tricep_extensions','lateral_shoulder_raises','jumping_jacks'}
BODY=(11,12,13,14,15,16,23,24,25,26,27,28)
TRIPLETS=((23,25,27),(24,26,28),(11,13,15),(12,14,16))
FEATURE_NAMES=tuple([f'position_{i}_{axis}' for i in range(33) for axis in ('x','y')]
                    +[f'velocity_{i}_{axis}' for i in range(33) for axis in ('x','y')]
                    +[f'quality_{i}' for i in range(33)]+[f'valid_{i}' for i in range(33)]
                    +[f'velocity_valid_{i}' for i in range(33)]
                    +['left_knee','right_knee','left_elbow','right_elbow']
                    +['left_knee_valid','right_knee_valid','left_elbow_valid','right_elbow_valid'])

def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def read_labels(path):
    rows=[]
    with Path(path).open(encoding='utf-8-sig',newline='') as handle:
        for number,row in enumerate(csv.reader(handle),1):
            if not row:continue
            if len(row)!=4:raise ValueError(f'{path}: row {number}: expected four columns')
            start,end,reps=map(int,row[:3]);name=row[3].strip()
            if start<0 or end<start or reps<0 or name not in ACTIVITIES:raise ValueError(f'{path}: invalid row {number}')
            rows.append(dict(start_frame=start,end_frame=end+1,original_activity=name,reference_reps=reps))
    rows.sort(key=lambda r:r['start_frame'])
    if any(a['end_frame']>b['start_frame'] for a,b in zip(rows,rows[1:])):raise ValueError('Overlapping CSV exercise intervals')
    return rows

def class_at(frame,labels):
    for row in labels:
        if row['start_frame']<=frame<row['end_frame']:
            name=row['original_activity'];return (name if name in CLASSES[:4] else 'other_exercise'),name
    return 'non_activity','non_activity'

def split_for(session,config):
    matches=[name for name,ids in config['splits'].items() if session in ids]
    if len(matches)!=1:raise ValueError(f'{session}: expected exactly one split role')
    return matches[0]

def valid_sample(frame,first,fps,hz):
    # Select the first source frame reaching each grid time. No future frame is used.
    if fps<hz:raise ValueError('Source FPS is lower than target rate; explicit resampling is required')
    return frame==first or math.floor((frame-first)*hz/fps+1e-8)>math.floor((frame-first-1)*hz/fps+1e-8)

class FeatureBuilder:
    """239 features; same class can be reused by a future streaming recognizer."""
    def __init__(self,quality=.5):self.quality=quality;self.previous=None;self.previous_mask=None;self.previous_time=None

    def update(self,landmarks,timestamp,width,height):
        points=np.asarray(landmarks,dtype=np.float32)
        if points.shape!=(33,5):raise ValueError('Expected 33 × [x,y,z,visibility,presence]')
        finite=np.isfinite(points).all(axis=1)
        quality=np.where(finite,np.minimum(points[:,3],points[:,4]),0).clip(0,1)
        valid=finite&(quality>=self.quality)&(points[:,0]>=0)&(points[:,0]<=1)&(points[:,1]>=0)&(points[:,1]<=1)
        pixels=points[:,:2]*np.array([width,height],np.float32)
        shoulders=[i for i in (11,12) if valid[i]];hips=[i for i in (23,24) if valid[i]]
        normalized=np.zeros((33,2),np.float32);scale=0.
        if shoulders and hips:
            centre=pixels[hips].mean(axis=0);scale=float(np.linalg.norm(pixels[shoulders].mean(axis=0)-centre))
            if scale>5:normalized[valid]=np.clip((pixels[valid]-centre)/scale,-8,8)
        pose_valid=scale>5 and sum(valid[list(BODY)])>=8
        mask=valid if scale>5 else np.zeros(33,bool)
        velocity=np.zeros_like(normalized);velocity_mask=np.zeros(33,bool)
        if self.previous_time is not None and 0<timestamp-self.previous_time<=.3:
            velocity_mask=mask&self.previous_mask
            velocity[velocity_mask]=np.clip((normalized[velocity_mask]-self.previous[velocity_mask])/(timestamp-self.previous_time),-20,20)
        values=np.zeros(4,np.float32);angle_mask=np.zeros(4,bool)
        for n,ids in enumerate(TRIPLETS):
            if all(valid[list(ids)]):
                result=angle(*pixels[list(ids)])
                if result is not None:values[n]=result/180;angle_mask[n]=True
        features=np.concatenate((normalized.ravel(),velocity.ravel(),quality,mask,velocity_mask,values,angle_mask)).astype(np.float32)
        self.previous=normalized;self.previous_mask=mask;self.previous_time=timestamp
        if not np.isfinite(features).all():raise ValueError('Nonfinite features')
        return features,bool(pose_valid)

def make_windows(frames,times,features,usable,labels,fps,config,background_reviewed=False):
    count=round(config['window_s']*config['target_hz']);stride=max(1,round(config['stride_s']*config['target_hz']))
    accepted=[];rejected={}
    def reject(reason):rejected[reason]=rejected.get(reason,0)+1
    for first in range(0,len(frames)-count+1,stride):
        end=first+count;f=frames[first:end];t=times[first:end]
        if np.any(np.diff(t)>1.6/config['target_hz']) or abs((t[-1]-t[0])-(count-1)/config['target_hz'])>1.1/fps:
            reject('time_discontinuity');continue
        identities=[class_at(int(i),labels) for i in f]
        if len(set(identities))!=1:reject('mixed_label');continue
        name,original=identities[0]
        if name=='non_activity':
            if not background_reviewed:reject('unreviewed_background');continue
            guard=config['background_guard_s']*fps
            if any(f[0]<r['end_frame']+guard and f[-1]>=r['start_frame']-guard for r in labels):reject('background_boundary');continue
        else:
            interval=next((r for r in labels if r['start_frame']<=f[0] and f[-1]<r['end_frame']),None)
            guard=config['label_guard_s']*fps
            if interval is None or f[0]<interval['start_frame']+guard or f[-1]>=interval['end_frame']-guard:reject('exercise_boundary');continue
        coverage=float(np.mean(usable[first:end]))
        if coverage<config['minimum_window_coverage']:reject('low_pose_coverage');continue
        accepted.append(dict(start_index=first,end_index=end,start_frame=int(f[0]),last_sample_frame=int(f[-1]),
                             start_s=float(t[0]),last_sample_s=float(t[-1]),class_name=name,class_id=CLASSES.index(name),
                             original_activity=original,pose_coverage=coverage))
    return accepted,rejected

def summarize_window(features):
    return np.concatenate((features.mean(axis=0),features.std(axis=0))).astype(np.float32)
