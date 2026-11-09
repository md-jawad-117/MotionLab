"""Exercise-specific signals with shared, inspectable cycle counting."""
import math
from .geometry import angle
from .motion import SquatMotion

EXERCISES={'squats':'Squats','pushups':'Push-ups','lunges':'Lunges','bicep_curls':'Bicep curls'}
JOINTS={'left_knee':(23,25,27),'right_knee':(24,26,28),
        'left_elbow':(11,13,15),'right_elbow':(12,14,16)}


def joint_measure(poses,ids,width,height):
    if not poses:
        return None,0.0,False
    points=[poses[0][i] for i in ids]
    if not all(all(math.isfinite(p[k]) for k in ('x','y','visibility','presence')) for p in points):
        return None,0.0,False
    value=angle(*[(p['x']*width,p['y']*height) for p in points])
    confidence=min(min(p['visibility'],p['presence']) for p in points)
    inside=all(0<=p['x']<=1 and 0<=p['y']<=1 for p in points)
    return value,confidence,inside


def angles_for(poses,width,height):
    result={}
    for name,ids in JOINTS.items():
        value,quality,inside=joint_measure(poses,ids,width,height)
        result[name+'_deg']=value
        result[name+'_quality']=quality
        result[name+'_valid']=value is not None and quality>=.5 and inside
    return result


class ArmMotion(SquatMotion):
    def __init__(self,exercise,side='both'):
        super().__init__(min_excursion=35/180,return_fraction=.3,min_rep_s=.4,max_rep_s=12)
        self.exercise=exercise;self.side=side

    def measure(self,poses,width,height):
        if not poses:
            return None,False,'missing_pose','none',False
        candidates=[]
        for side,ids in [('left',(11,13,15)),('right',(12,14,16))]:
            if self.side!='both' and side!=self.side:
                continue
            value,quality,inside=joint_measure(poses,ids,width,height)
            if value is not None:
                candidates.append((side,value,max(.05,quality),inside and quality>=.5))
        if not candidates:
            return None,False,'missing_elbow_geometry','none',False
        good=[c for c in candidates if c[3]]
        used=good or candidates
        weight=sum(c[2] for c in used)
        elbow=sum(c[1]*c[2] for c in used)/weight
        reliable=bool(good)
        reason='' if reliable else 'low_quality_arm'
        if self.exercise=='pushups':
            orientations=[]
            for ids in ((11,23,27),(12,24,28)):
                points=[poses[0][i] for i in ids]
                shoulder,hip,ankle=points
                dx=(ankle['x']-shoulder['x'])*width
                dy=(ankle['y']-shoulder['y'])*height
                distance=math.hypot(dx,dy)
                if distance>30 and min(min(p['visibility'],p['presence']) for p in points)>=.4:
                    orientations.append(abs(dy)/distance<.75)
            if not any(orientations):
                reliable=False;reason='pushup_body_view_unconfirmed'
        return (180-elbow)/180,reliable,reason,'+'.join(c[0] for c in used),elbow>=145


class LungeMotion(SquatMotion):
    def __init__(self):
        super().__init__(min_excursion=.06,min_rep_s=.5,max_rep_s=12)

    def measure(self,poses,width,height):
        if not poses:
            return None,False,'missing_pose','none',False
        pose=poses[0];chains=[]
        for side,ids in [('left',(11,23,27)),('right',(12,24,28))]:
            points=[pose[i] for i in ids]
            if not all(all(math.isfinite(p[k]) for k in ('x','y','visibility','presence')) for p in points):
                continue
            q=min(min(p['visibility'],p['presence']) for p in points)
            valid=q>=.5 and all(0<=p['x']<=1 and 0<=p['y']<=1 for p in points)
            if points[2]['y']>points[0]['y']+.03:
                chains.append((side,max(q,.05),points,valid))
        if not chains:
            return None,False,'missing_body_geometry','none',False
        good=[c for c in chains if c[3]];used=good or chains
        weight=sum(c[1] for c in used)
        shoulder=sum(c[1]*c[2][0]['y'] for c in used)/weight*height
        hip=sum(c[1]*c[2][1]['y'] for c in used)/weight*height
        floor=max(c[2][2]['y'] for c in used)*height
        span=floor-shoulder
        if span<=20 or not shoulder<hip<floor:
            return None,False,'degenerate_body_geometry','none',False
        if self.scale is None:
            if not good:
                return None,False,'calibration_pending','none',False
            self.scale=span
        value=-(floor-hip)/self.scale
        return value,bool(good),'' if good else 'low_quality_body_chain','+'.join(c[0] for c in used),(floor-hip)/span>=.5


class ExerciseAnalyzer:
    """No labels/reference counts enter these signal or counting rules."""
    def __init__(self,exercise,start_s,end_s,side='both',curl_policy='per_arm'):
        if exercise not in EXERCISES:
            raise ValueError(f'Unsupported exercise: {exercise}')
        self.exercise=exercise
        self.curl_policy=curl_policy
        self._pair_pending={m:None for m in ('raw','smoothed','gated')}
        if exercise=='bicep_curls':
            names=('left','right') if side=='both' else (side,)
            self.engines={s:ArmMotion(exercise,s) for s in names}
        else:
            engine=SquatMotion() if exercise=='squats' else LungeMotion() if exercise=='lunges' else ArmMotion(exercise,side)
            self.engines={'body' if exercise in ('squats','lunges') else 'arms':engine}
        for engine in self.engines.values():
            engine.set_window(start_s,end_s)
        self.events=[];self._counts={m:0 for m in ('raw','smoothed','gated')}
        self._last_event={m:None for m in self._counts}

    def update(self,poses,frame,timestamp,width,height):
        row=dict(frame=frame,timestamp_s=timestamp,exercise=self.exercise,**angles_for(poses,width,height))
        outputs=[]
        for name,engine in self.engines.items():
            output=engine.update(poses,timestamp,width,height)
            outputs.append(output)
            for event in engine.events:
                event['side']=name
            pending=engine.events[getattr(engine,'_reported_events',0):]
            engine._reported_events=len(engine.events)
            for event in pending:
                if not event['assigned_to_window']:
                    continue
                mode=event['mode']
                if self.exercise=='bicep_curls' and self.curl_policy=='paired' and len(self.engines)>1:
                    waiting=self._pair_pending[mode]
                    if waiting is None or waiting['side']==event['side'] or event['completion_s']-waiting['completion_s']>6:
                        self._pair_pending[mode]=dict(event)
                        continue
                    event=dict(event,start_s=min(waiting['start_s'],event['start_s']),
                               side='both',amplitude=(waiting['amplitude']+event['amplitude'])/2)
                    self._pair_pending[mode]=None
                previous=self._last_event[mode]
                # A bilateral simultaneous curl is one shared cycle; alternating
                # complete arm cycles remain separate. Policy is recorded in reports.
                paired=self.exercise=='bicep_curls' and self.curl_policy=='per_arm' and previous is not None and event['side']!=previous['side'] and abs(event['completion_s']-previous['completion_s'])<=.25
                if paired:
                    previous['side']='both'
                    continue
                copied=dict(event,exercise=self.exercise,rep_index=self._counts[mode]+1)
                self._counts[mode]+=1;self.events.append(copied);self._last_event[mode]=copied
        for mode in self._counts:
            valid=[o['motion_'+mode+'_value'] for o in outputs if o['motion_'+mode+'_value'] is not None]
            row[mode+'_signal']=sum(valid)/len(valid) if valid else None
            row[mode+'_reps']=self._counts[mode]
        row['tracking_valid']=any(o['motion_reliable'] for o in outputs)
        row['support']=';'.join(o['motion_support'] for o in outputs)
        row['phase']=' / '.join(o['motion_phase'] for o in outputs)
        row['rejection_reasons']=';'.join(sorted(set(o['motion_rejection_reasons'] for o in outputs if o['motion_rejection_reasons'])))
        return row
