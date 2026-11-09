"""Local workout sessions, explicit exercise segments and per-repetition data."""
import base64,csv,hashlib,html,json,math,time
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from .exercises import EXERCISES,ExerciseAnalyzer
from .pose import PoseEstimator
from .video_io import VideoReader,writer
from .visualize import EDGES


class Cancelled(Exception):
    pass


def probe_video(path):
    reader=VideoReader(path)
    try:
        frames=int(reader.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frames<=0:
            raise ValueError('Video duration could not be read. Use a constant-frame-rate MP4.')
        return dict(fps=reader.fps,frames=frames,width=reader.width,height=reader.height,duration_s=frames/reader.fps)
    finally:
        reader.close()


def import_labels(path,fps,total_frames):
    segments=[];skipped=[]
    with open(path,newline='',encoding='utf-8-sig') as handle:
        for line,row in enumerate(csv.reader(handle),1):
            if not row or not any(x.strip() for x in row):
                continue
            if row[0].strip().lower() in ('start','start_frame','start frame'):
                continue
            if len(row)<4:
                raise ValueError(f'Label row {line} needs start frame, end frame, count and exercise.')
            try:
                first,last,count=map(int,row[:3])
            except ValueError as error:
                raise ValueError(f'Invalid numbers in label row {line}.') from error
            exercise=row[3].strip()
            if exercise not in EXERCISES:
                skipped.append(exercise);continue
            if not 0<=first<=last<total_frames or count<0:
                raise ValueError(f'Label row {line} is outside this video or has an invalid count.')
            segments.append(dict(exercise=exercise,first_frame=first,end_frame=last+1,side='both',curl_policy='per_arm',reference=count,
                                 label_source=str(Path(path).resolve())))
    return segments,skipped


def validate_segments(segments,meta):
    normalized=[]
    for number,segment in enumerate(segments,1):
        item=dict(segment)
        if item.get('exercise') not in EXERCISES:
            raise ValueError(f'Segment {number}: choose a supported exercise.')
        first=int(item['first_frame']);end=int(item['end_frame'])
        if not 0<=first<end<=meta['frames']:
            raise ValueError(f'Segment {number}: start/end must be inside the video, with end after start.')
        if (end-first)/meta['fps']<.5:
            raise ValueError(f'Segment {number}: select at least half a second.')
        if item.get('side','both') not in ('both','left','right'):
            raise ValueError(f'Segment {number}: invalid arm selection.')
        if item.get('curl_policy','per_arm') not in ('per_arm','paired'):
            raise ValueError(f'Segment {number}: invalid curl repetition convention.')
        if item.get('reference') is not None and int(item['reference'])<0:
            raise ValueError('Reference counts must be nonnegative.')
        item.update(first_frame=first,end_frame=end,side=item.get('side','both'))
        normalized.append(item)
    normalized.sort(key=lambda s:s['first_frame'])
    for previous,current in zip(normalized,normalized[1:]):
        if previous['end_frame']>current['first_frame']:
            raise ValueError('Exercise segments overlap. Adjust the timeline so each frame belongs to one exercise.')
    if not normalized:
        raise ValueError('Add at least one exercise segment.')
    return normalized


def save_json(path,value):
    temp=Path(str(path)+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(path)


def write_csv(path,rows):
    with open(path,'w',newline='',encoding='utf-8-sig') as handle:
        if rows:
            table=csv.DictWriter(handle,fieldnames=list(rows[0]))
            table.writeheader();table.writerows(rows)


def annotate(frame,poses,row,exercise):
    height,width=frame.shape[:2]
    if poses:
        pose=poses[0]
        for a,b in EDGES:
            if all(min(pose[i]['visibility'],pose[i]['presence'])>=.5 for i in (a,b)):
                points=[(int(pose[i]['x']*width),int(pose[i]['y']*height)) for i in (a,b)]
                cv2.line(frame,*points,(50,230,145),max(1,round(width/600)),cv2.LINE_AA)
    joint='elbow' if exercise in ('pushups','bicep_curls') else 'knee'
    labels=[]
    for side in ('left','right'):
        value=row[side+'_'+joint+'_deg']
        labels.append(side.title()+': '+('--' if value is None else f'{value:.1f} deg'))
    lines=[f"{EXERCISES[exercise]} | reps {row['gated_reps']} | {row['timestamp_s']:.2f}s"+(' | CONTEXT' if not row['in_score_window'] else ''),
           '    '.join(labels),
           'Tracking: '+('available' if row['tracking_valid'] else row['rejection_reasons'] or 'limited')+' | '+row['support']]
    unit=width/1280;step=max(18,round(31*unit));margin=max(6,round(12*unit));scale=max(.34,.72*unit)
    cv2.rectangle(frame,(0,0),(width,min(height,step*3+margin)),(20,29,42),-1)
    for i,text in enumerate(lines):
        size=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,scale,1)[0][0]
        fit=min(scale,scale*(width-2*margin)/max(size,1))
        cv2.putText(frame,text,(margin,step*(i+1)),cv2.FONT_HERSHEY_SIMPLEX,fit,
                    (235,248,250) if i!=1 else (145,235,205),max(1,round(unit)),cv2.LINE_AA)
    return frame


def per_rep_analysis(events,rows,exercise):
    joint='elbow' if exercise in ('pushups','bicep_curls') else 'knee'
    result=[]
    for index,event in enumerate((e for e in events if e['mode']=='gated'),1):
        cycle=[r for r in rows if event['start_s']<=r['timestamp_s']<=event['completion_s']]
        if not cycle:
            continue
        ranges={};means=[]
        for side in ('left','right'):
            values=[r[side+'_'+joint+'_deg'] for r in cycle if r[side+'_'+joint+'_valid']]
            ranges[side+'_range_deg']=max(values)-min(values) if values else None
            if ranges[side+'_range_deg'] is not None and (event['side'] in ('body','arms','both',side)):
                means.append(ranges[side+'_range_deg'])
        coverage=sum(r['tracking_valid'] for r in cycle)/len(cycle)
        flags=[]
        if coverage<.9:
            flags.append('tracking gaps')
        if not means:
            flags.append('joint range unavailable')
        result.append(dict(rep=index,exercise=exercise,side=event['side'],start_s=event['start_s'],
                           turning_s=event['bottom_s'],end_s=event['completion_s'],
                           duration_s=event['completion_s']-event['start_s'],
                           lowering_or_flexion_s=event['bottom_s']-event['start_s'],
                           return_s=event['completion_s']-event['bottom_s'],
                           range_deg=float(np.mean(means)) if means else None,**ranges,
                           signal_excursion=event['amplitude'],tracking_coverage=coverage,
                           flags='; '.join(flags)))
    return result


def consistency(values):
    values=[v for v in values if v is not None and math.isfinite(v)]
    if not values:
        return dict(mean=None,minimum=None,maximum=None,cv_percent=None)
    mean=float(np.mean(values))
    return dict(mean=mean,minimum=min(values),maximum=max(values),
                cv_percent=float(np.std(values)/mean*100) if len(values)>1 and mean>0 else None)


def make_plots(folder,rows,reps,exercise):
    # Non-GUI rendering avoids creating Qt objects on the processing thread.
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig=Figure(figsize=(11,6),layout='constrained');FigureCanvasAgg(fig)
    axes=fig.subplots(2,1,sharex=True)
    times=[r['timestamp_s'] for r in rows]
    for mode in ('raw','smoothed','gated'):
        axes[0].plot(times,np.array([r[mode+'_signal'] for r in rows],float),label=mode)
        axes[1].plot(times,[r[mode+'_reps'] for r in rows],label=mode)
    axes[0].set(title=EXERCISES[exercise]+' — measured movement',ylabel='Normalized movement signal')
    axes[1].set(xlabel='Source time (s)',ylabel='Detected repetitions')
    for axis in axes:
        axis.legend();axis.grid(alpha=.2)
    fig.savefig(folder/'analysis.png',dpi=130)


def validate_crop(crop,meta):
    if crop is None:return None
    if not isinstance(crop,(list,tuple)) or len(crop)!=4:
        raise ValueError('Crop must contain source-pixel x, y, width and height.')
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or int(v)!=v for v in crop):
        raise ValueError('Crop coordinates must be finite whole pixels.')
    x,y,width,height=map(int,crop)
    if x<0 or y<0 or width<64 or height<64 or x+width>meta['width'] or y+height>meta['height']:
        raise ValueError('Crop must stay inside the source video and be at least 64 × 64 pixels.')
    return [x,y,width,height]


def run_workout(video,segments,output,progress=None,preview=None,cancelled=None,replay_session=None,crop=None,timeline=None):
    started=time.perf_counter();video=Path(video).resolve();out=Path(output).resolve()
    meta=probe_video(video);segments=validate_segments(segments,meta);crop=validate_crop(crop,meta)
    timeline=validate_segments(timeline,meta) if timeline is not None else None
    bounds=lambda s:(s['exercise'],s['first_frame'],s['end_frame'])
    if timeline is not None and any(bounds(s) not in {bounds(t) for t in timeline} for s in segments):
        raise ValueError('Selected exercise sections must belong to the supplied timeline.')
    project=Path(__file__).resolve().parents[2]
    model=project/'assets/pose_landmarker_lite.task'
    if not model.is_file():
        raise ValueError('Pose model is missing. Run scripts/download_model.py once from the project folder.')
    replay=None;verified_hash=None
    if replay_session:
        replay=json.loads(Path(replay_session).read_text(encoding='utf-8'))
        if replay.get('crop')!=crop:
            raise ValueError('Replay requires the original crop. Run fresh inference for a changed crop.')
        digest=hashlib.sha256()
        with video.open('rb') as handle:
            for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
        verified_hash=digest.hexdigest()
        if verified_hash!=replay.get('video_sha256') or replay['metadata']!=meta:
            raise ValueError('Replay source hash or video metadata does not match.')
        if len(replay['segments'])!=len(segments) or any((a['exercise'],a['first_frame'],a['end_frame'])!=(b['exercise'],b['first_frame'],b['end_frame']) for a,b in zip(replay['segments'],segments)):
            raise ValueError('Replay requires matching exercise types and source intervals; counting policies may change.')
    out.mkdir(parents=True,exist_ok=False)
    session=dict(schema=1,video=str(video),metadata=meta,created=datetime.now().isoformat(timespec='seconds'),
                 engine='motionlab-desktop-v1',replay_source=str(Path(replay_session).resolve()) if replay_session else None,
                 status='running',segments=segments,timeline=timeline,results=[],crop=crop,
                 landmark_coordinates='Normalized to cropped/resized analysis image; crop uses original source pixels [x,y,width,height].',
                 segmentation='User-selected or imported exercise boundaries; no automatic exercise recognition.',
                 rep_policy='One complete cycle for squats/push-ups/lunges. Curls follow each segment\'s selected policy: per-arm cycles (near-simultaneous bilateral cycles merge), or one left/right pair. A selected single arm always counts its own cycles. Check the convention against any supplied reference.',
                 limits='Single-person, static-camera prototype. Angles are 2D projections; no medical/form-safety certification. Exercise labels are supplied by the user or CSV.')
    save_json(out/'session.json',session)
    ranges=[]
    context_timeline=timeline if timeline is not None else segments
    for s in segments:
        i=next(i for i,t in enumerate(context_timeline) if bounds(t)==bounds(s))
        first=max(0,s['first_frame']-round(2*meta['fps']))
        last=min(meta['frames'],s['end_frame']+round(meta['fps']))
        if i:
            first=max(first,context_timeline[i-1]['end_frame'])
        if i+1<len(context_timeline):
            last=min(last,context_timeline[i+1]['first_frame'])
        ranges.append((first,last))
    total=sum(last-first for first,last in ranges);processed=0;all_reps=[]
    try:
        for number,(segment,(first,last)) in enumerate(zip(segments,ranges),1):
            if cancelled and cancelled():
                raise Cancelled()
            exercise=segment['exercise'];folder=out/f'{number:02d}_{exercise}';folder.mkdir()
            analyzer=ExerciseAnalyzer(exercise,segment['first_frame']/meta['fps'],segment['end_frame']/meta['fps'],segment['side'],segment.get('curl_policy','per_arm'))
            reader=VideoReader(video);pose=encoder=None;rows=[];inference=0.0
            input_width,input_height=crop[2:] if crop else (meta['width'],meta['height'])
            factor=min(1,1280/max(input_width,input_height))
            width=max(2,int(input_width*factor)//2*2);height=max(2,int(input_height*factor)//2*2)
            cached=None
            if replay:
                old=replay['results'][number-1]
                if (old['source_first_frame'],old['source_end_frame'],old['render_width'],old['render_height'])!=(first,last,width,height):
                    raise ValueError('Replay context or render dimensions changed; run fresh analysis.')
                with (Path(old['directory'])/'landmarks.jsonl').open(encoding='utf-8') as handle:
                    cached={r['frame']:r for r in map(json.loads,handle)}
            try:
                if cached is None:pose=PoseEstimator(model)
                encoder=writer(folder/'annotated.mp4',meta['fps'],width,height)
                with (folder/'landmarks.jsonl').open('w',encoding='utf-8') as landmarks:
                    for frame_index,timestamp,image in reader.frames(first,last):
                        if cancelled and cancelled():
                            raise Cancelled()
                        if crop:
                            x,y,crop_width,crop_height=crop;image=image[y:y+crop_height,x:x+crop_width]
                        image=cv2.resize(image,(width,height))
                        if cached is None:
                            poses=pose.detect(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),round(timestamp*1000))
                        else:
                            if frame_index not in cached or abs(cached[frame_index]['timestamp_s']-timestamp)>1e-6:
                                raise ValueError('Replay is missing a required context frame or timestamp.')
                            poses=cached[frame_index]['poses']
                        landmarks.write(json.dumps(dict(frame=frame_index,timestamp_s=timestamp,poses=poses))+'\n')
                        row=analyzer.update(poses,frame_index,timestamp,width,height)
                        row['in_score_window']=segment['first_frame']<=frame_index<segment['end_frame']
                        rows.append(row);processed+=1
                        image=annotate(image,poses,row,exercise);encoder.write(image)
                        if frame_index==first:
                            cv2.imwrite(str(folder/'preview.jpg'),image)
                        if progress and processed%10==0:
                            progress(round(processed/total*100),f'{number}/{len(segments)} · {EXERCISES[exercise]} · {row["gated_reps"]} reps')
                        if preview and processed%15==0:
                            preview_scale=min(640/width,360/height,1)
                            small=cv2.resize(image,(max(2,round(width*preview_scale)),max(2,round(height*preview_scale))))
                            preview(cv2.cvtColor(small,cv2.COLOR_BGR2RGB))
                inference=pose.inference_s if pose is not None else None
            finally:
                reader.close()
                if pose is not None:pose.close()
                if encoder is not None:encoder.release()
            if len(rows)!=last-first:
                raise ValueError('The video ended before the requested segment. Check that its download is complete.')
            reps=per_rep_analysis(analyzer.events,rows,exercise)
            score=[r for r in rows if r['in_score_window']]
            count=rows[-1]['gated_reps']
            timing=consistency([r['duration_s'] for r in reps]);ranges_stats=consistency([r['range_deg'] for r in reps])
            coverage=sum(r['tracking_valid'] for r in score)/len(score)
            feedback=[]
            if not count:feedback.append('No complete cycles detected. Review the exercise selection, segment boundaries and visible joints.')
            if coverage<.9:feedback.append(f'Tracking was available for {coverage:.0%} of this segment. Review gaps before interpreting counts or ranges.')
            if timing['cv_percent'] is not None:feedback.append(f'Detected rep timing variation: {timing["cv_percent"]:.1f}% CV (lower means more uniform timing in this clip).')
            if ranges_stats['cv_percent'] is not None:feedback.append(f'Observed 2D joint-range variation: {ranges_stats["cv_percent"]:.1f}% CV; viewpoint and tracking affect this measurement.')
            result=dict(segment=number,exercise=exercise,title=EXERCISES[exercise],count=count,reference=segment.get('reference'),
                        curl_policy=segment.get('curl_policy','per_arm'),arm_selection=segment['side'],
                        reference_error=abs(count-int(segment['reference'])) if segment.get('reference') is not None else None,
                        directory=str(folder),source_first_frame=first,source_end_frame=last,
                        start_s=segment['first_frame']/meta['fps'],end_s=segment['end_frame']/meta['fps'],
                        duration=timing,range=ranges_stats,tracking_coverage=coverage,feedback=feedback,
                        reps=reps,render_width=width,render_height=height,crop=crop,inference_s=inference,
                        inference_fps=len(rows)/inference if inference else None,
                        mode_counts={m:rows[-1][m+'_reps'] for m in ('raw','smoothed','gated')})
            write_csv(folder/'measurements.csv',rows);write_csv(folder/'reps.csv',reps)
            save_json(folder/'events.json',analyzer.events);save_json(folder/'summary.json',result)
            make_plots(folder,rows,reps,exercise)
            session['results'].append(result)
            all_reps.extend(dict(segment=number,**rep) for rep in reps)
            save_json(out/'session.json',session);write_csv(out/'all_reps.csv',all_reps)
        session['status']='complete';session['end_to_end_s']=time.perf_counter()-started
        if verified_hash is None:
            digest=hashlib.sha256()
            with video.open('rb') as handle:
                for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
            verified_hash=digest.hexdigest()
        session['video_sha256']=verified_hash
        save_json(out/'session.json',session)
        export_report(out,session)
        if progress:progress(100,'Analysis complete')
        return out/'session.json'
    except Cancelled:
        session['status']='cancelled';save_json(out/'session.json',session)
        export_report(out,session)
        raise
    except Exception as error:
        session.update(status='error',error=str(error));save_json(out/'session.json',session)
        raise


def export_report(out,session,filename='report.html'):
    esc=html.escape
    body=['<!doctype html><html><meta charset="utf-8"><title>MotionLab workout report</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto;color:#172c42;padding:0 24px}h1{color:#087f78}table{border-collapse:collapse;width:100%;margin:20px 0}th,td{text-align:left;padding:10px;border-bottom:1px solid #dde5eb}th{background:#edf7f5}.note{color:#536779}img{max-width:100%}</style><h1>MotionLab · Workout analysis</h1>',
          '<p>'+esc(Path(session['video']).name)+' · '+esc(session['status'])+'</p>',
          '<p class="note">'+esc(session['segmentation'])+'</p>',
          '<table><tr><th>Segment</th><th>Exercise</th><th>Detected</th><th>Reference</th><th>Tracking</th><th>Mean rep duration</th></tr>']
    for r in session['results']:
        mean=r['duration']['mean'];duration='—' if mean is None else f'{mean:.2f} s'
        body.append(f'<tr><td>{r["segment"]}</td><td>{esc(r["title"])}</td><td>{r["count"]}</td><td>{r["reference"] if r["reference"] is not None else "—"}</td><td>{r["tracking_coverage"]:.0%}</td><td>{duration}</td></tr>')
    body.append('</table>')
    if session.get('crop'):
        body.append('<p>Analysis crop in source pixels [x, y, width, height]: '+esc(str(session['crop']))+'. Landmarks and annotated clips use this cropped image.</p>')
    for r in session['results']:
        directory=Path(r['directory'])
        body.append('<h2>'+esc(f'{r["segment"]}. {r["title"]}')+'</h2>')
        body.extend('<p>'+esc(message)+'</p>' for message in r['feedback'])
        if (directory/'analysis.png').exists():
            encoded=base64.b64encode((directory/'analysis.png').read_bytes()).decode('ascii')
            body.append(f'<img src="data:image/png;base64,{encoded}">')
        annotated=Path(session['annotated_video']) if session.get('classifier') else directory/'annotated.mp4'
        body.append(f'<p><a href="{annotated.as_uri()}">Annotated video in project</a> · <a href="{(directory/"reps.csv").as_uri()}">Per-rep CSV in project</a></p>')
        body.append('<table><tr><th>Rep</th><th>Side</th><th>Duration (s)</th><th>Observed range (°)</th><th>Tracking</th></tr>')
        for rep in r['reps']:
            observed='—' if rep['range_deg'] is None else f'{rep["range_deg"]:.1f}'
            body.append(f'<tr><td>{rep["rep"]}</td><td>{esc(rep["side"])}</td><td>{rep["duration_s"]:.2f}</td><td>{observed}</td><td>{rep["tracking_coverage"]:.0%}</td></tr>')
        body.append('</table>')
    body.extend(['<h2>Counting convention and limits</h2><p>'+esc(session['rep_policy'])+'</p>',
                 '<p class="note">'+esc(session['limits'])+'</p>',
                 '<p>Rep duration is the detected movement onset to observed return. Range is the observed 2D joint-angle excursion, not calibrated anatomical ROM. Only cycles with a turning point inside the labeled segment are assigned; explicit context may confirm their return. CV measures variability, not safe or correct form.</p></html>'])
    (Path(out)/filename).write_text('\n'.join(body),encoding='utf-8')
