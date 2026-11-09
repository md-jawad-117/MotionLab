"""Causal squat-motion measurements, independent of reference counts."""
import math
from collections import deque
from .filters import CausalEMA


class ExcursionCounter:
    """Count down/up hip excursions after observing an upright anchor.

    Input increases as the hips descend. Only observed return crossings count;
    missing frames are never filled. A long gap abandons a pending cycle.
    """
    def __init__(self, min_excursion=0.08, return_fraction=0.25,
                 min_rep_s=0.6, max_rep_s=8.0, reset_gap_s=0.4):
        self.excursion, self.return_fraction = min_excursion, return_fraction
        self.minimum, self.maximum, self.gap = min_rep_s, max_rep_s, reset_gap_s
        self.count=0
        self.last=None
        self.reset()

    def reset(self):
        self.state='unarmed'
        self.top=self.bottom=self.started=self.bottom_time=None

    def update(self,value,timestamp,upright=False):
        event=None
        if self.last is not None and timestamp-self.last>self.gap:
            self.reset()
        if value is None or not math.isfinite(value):
            return self.count,event
        self.last=timestamp
        if self.state=='unarmed':
            if upright:
                self.state='standing';self.top=value;self.started=timestamp
            return self.count,event
        if self.state=='standing':
            if value<self.top:
                self.top=value;self.started=timestamp
            if value-self.top<0.02:
                self.started=timestamp
            if value-self.top>=self.excursion:
                self.state='bottom';self.bottom=value;self.bottom_time=timestamp
        else:
            if value>self.bottom:
                self.bottom=value;self.bottom_time=timestamp
            if timestamp-self.started>self.maximum:
                self.reset()
                return self.count,event
            amplitude=self.bottom-self.top
            if value<=self.top+self.return_fraction*amplitude:
                if timestamp-self.started>=self.minimum:
                    self.count+=1
                    event=dict(start_s=self.started,bottom_s=self.bottom_time,
                               completion_s=timestamp,amplitude=amplitude)
                self.state='standing';self.top=value;self.started=timestamp
        return self.count,event


class SquatMotion:
    def __init__(self,min_confidence=0.5,time_constant_s=0.06,reset_gap_s=0.4,
                 min_excursion=0.08,return_fraction=0.25,min_rep_s=0.6,max_rep_s=8.0,
                 max_speed_per_s=4.0):
        self.confidence=min_confidence
        self.scale=None
        self.max_speed=max_speed_per_s
        self.previous=self.previous_time=None
        self.window=(None,None)
        self.score={m:0 for m in ('raw','smoothed','gated')}
        self.strict={m:0 for m in ('raw','smoothed','gated')}
        self.events=[]
        self.history=deque()
        self.history_seconds=max_rep_s+reset_gap_s+1
        self.filters={m:CausalEMA(time_constant_s,reset_gap_s) for m in ('smoothed','gated')}
        self.counters={m:ExcursionCounter(min_excursion,return_fraction,min_rep_s,max_rep_s,reset_gap_s)
                       for m in ('raw','smoothed','gated')}

    def measure(self,poses,width,height):
        if not poses:
            return None,False,'missing_pose','none',False
        pose=poses[0]
        # Prefer visible same-side chains; retain low-quality raw values for comparison.
        chains=[]
        for side,ids in [('left',(11,23,27)),('right',(12,24,28))]:
            points=[pose[i] for i in ids]
            finite=all(all(math.isfinite(p[k]) for k in ('x','y','visibility','presence')) for p in points)
            if not finite:
                continue
            quality=min(min(p['visibility'],p['presence']) for p in points)
            in_frame=all(0<=p['x']<=1 and 0<=p['y']<=1 for p in points)
            shoulder,hip,ankle=[p['y']*height for p in points]
            span=ankle-shoulder
            if span>20 and shoulder<hip<ankle:
                chains.append((side,max(quality,0.05),ankle-hip,span,in_frame and quality>=self.confidence))
        if not chains:
            return None,False,'no_visible_body_chain','none',False
        good=[c for c in chains if c[4]]
        used=good or chains
        weight=sum(c[1] for c in used)
        leg_height=sum(c[1]*c[2] for c in used)/weight
        span=sum(c[1]*c[3] for c in used)/weight
        if self.scale is None:
            if not good:
                return None,False,'calibration_pending','none',False
            self.scale=span
        value=-leg_height/self.scale
        upright=leg_height/span>=0.52
        return value,bool(good),'' if good else 'low_quality_body_chain','+'.join(c[0] for c in used),upright

    def set_window(self,start_s=None,end_s=None):
        self.window=(start_s,end_s)

    def inside(self,timestamp):
        start,end=self.window
        return (start is None or timestamp>=start) and (end is None or timestamp<end)

    def update(self,poses,timestamp,width,height):
        raw,reliable,reasons,support,upright=self.measure(poses,width,height)
        if reliable and self.previous_time is not None and timestamp-self.previous_time<=0.4:
            dt=timestamp-self.previous_time
            if dt<=0 or abs(raw-self.previous)/dt>self.max_speed:
                reliable=False;reasons='body_motion_jump'
        if reliable:
            self.previous,self.previous_time=raw,timestamp
        if raw is not None:
            self.history.append((timestamp,raw,reliable))
        while self.history and timestamp-self.history[0][0]>self.history_seconds:
            self.history.popleft()
        values=dict(raw=raw,smoothed=self.filters['smoothed'].update(raw,timestamp),
                    gated=self.filters['gated'].update(raw if reliable else None,timestamp))
        row=dict(motion_reliable=reliable,motion_rejection_reasons=reasons,motion_support=support,
                 motion_upright_hint=upright,motion_scale_px=self.scale)
        for mode,value in values.items():
            count,event=self.counters[mode].update(value,timestamp,upright)
            if event is not None:
                # Use the source-signal peak inside this detected cycle for
                # membership. Filtering can move its own peak across a label edge.
                candidates=[h for h in self.history if event['start_s']<=h[0]<=timestamp
                            and (mode!='gated' or h[2])]
                event['filtered_bottom_s']=event['bottom_s']
                if candidates:
                    event['bottom_s']=max(candidates,key=lambda h:h[1])[0]
                assigned=self.inside(event['bottom_s'])
                self.events.append(dict(mode=mode,assigned_to_window=assigned,**event))
                if assigned:
                    self.score[mode]+=1
                if self.inside(event['completion_s']):
                    self.strict[mode]+=1
            row['motion_'+mode+'_value']=value
            row['motion_'+mode+'_reps']=self.score[mode]
            row['motion_'+mode+'_strict_reps']=self.strict[mode]
            row['motion_'+mode+'_event']=event is not None and self.inside(event['bottom_s'])
            row['motion_'+mode+'_bottom_s']=event['bottom_s'] if event is not None else None
        row['motion_phase']=self.counters['gated'].state
        return row
