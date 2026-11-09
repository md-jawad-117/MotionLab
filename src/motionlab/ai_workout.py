"""Automatic video demonstration using MediaPipe and a frozen exercise classifier."""
import csv
from datetime import datetime
import json
from pathlib import Path
import time
import cv2
import numpy as np
from .exercises import EXERCISES,ExerciseAnalyzer,angles_for
from .ml_data import FeatureBuilder,valid_sample,sha256
from .pose import PoseEstimator
from .recognition import ExerciseClassifier,RecognitionStream,TITLES
from .video_io import VideoReader,writer
from .visualize import EDGES
from .workout import (Cancelled,probe_video,save_json,write_csv,per_rep_analysis,
    consistency,make_plots,export_report)

PROJECT=Path(__file__).resolve().parents[2]


def overlay(image,poses,decision,angles,counts,timestamp):
    height,width=image.shape[:2]
    if poses:
        for a,b in EDGES:
            points=[poses[0][i] for i in (a,b)]
            if all(min(p['visibility'],p['presence'])>=.5 and 0<=p['x']<=1 and 0<=p['y']<=1 for p in points):
                coords=[(round(p['x']*(width-1)),round(p['y']*(height-1))) for p in points]
                cv2.line(image,*coords,(65,235,160),max(1,round(width/600)),cv2.LINE_AA)
    label=decision['stable_class'] if decision['status']=='ready' else decision['raw_class']
    title=TITLES.get(label,'Waiting for visible movement')
    if decision['status']=='confirming':title='Confirming: '+title
    if decision['status']=='warming_up':title='AI warm-up: collecting ~3 seconds'
    if decision['status']=='tracking_limited':title='AI: tracking limited'
    score='' if decision['raw_score'] is None else f' | model score {decision["raw_score"]:.0%}'
    def joint(name):
        return '--' if not angles.get(name+'_valid') else f'{angles[name+"_deg"]:.1f}'
    count=f'Rep estimate: {counts.get(label,0)} since recognition' if label in EXERCISES else 'Rep counting paused'
    lines=[f'{title}{score}',
        f'Knees: L {joint("left_knee")} deg   R {joint("right_knee")} deg  |  '
        f'Elbows: L {joint("left_elbow")} deg   R {joint("right_elbow")} deg',
        f'{count}  |  {timestamp:.2f}s  |  '+decision['status'].replace('_',' ')]
    scale=max(.35,.66*width/1280);step=max(19,round(30*width/1280));margin=8
    cv2.rectangle(image,(0,0),(width,min(height,step*3+margin)),(23,36,46),-1)
    for i,text in enumerate(lines):
        extent=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,scale,1)[0][0]
        fit=scale*min(1,(width-16)/max(extent,1))
        cv2.putText(image,text,(8,step*(i+1)),cv2.FONT_HERSHEY_SIMPLEX,fit,
            (230,245,245) if i!=1 else (140,235,205),1,cv2.LINE_AA)
    return image


def timeline_plot(output,predictions):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from .ml_data import CLASSES
    figure=Figure(figsize=(11,5),layout='constrained');FigureCanvasAgg(figure)
    axes=figure.subplots(2,1,sharex=True)
    times=[r['timestamp_s'] for r in predictions]
    axes[0].step(times,[CLASSES.index(r['stable_class']) if r['stable_class'] else np.nan for r in predictions],where='post')
    axes[0].set(yticks=range(6),yticklabels=[TITLES[k] for k in CLASSES],ylabel='Confirmed AI class')
    axes[1].plot(times,[r['raw_score'] if r['raw_score'] is not None else np.nan for r in predictions],label='Uncalibrated model score')
    axes[1].plot(times,[r['window_coverage'] for r in predictions],label='Usable pose coverage')
    axes[1].set(xlabel='Source time (s)',ylim=(0,1.05));axes[1].legend(fontsize=8)
    for axis in axes:axis.grid(alpha=.2)
    figure.savefig(output/'ai_timeline.png',dpi=140)


def run_auto(video,output,progress=None,preview=None,cancelled=None,assets=None):
    started=time.perf_counter();video=Path(video).resolve();out=Path(output).resolve()
    meta=probe_video(video)
    if meta['fps']<15:raise ValueError('Automatic AI mode needs a video of at least 15 FPS.')
    assets=Path(assets or PROJECT/'assets')
    classifier=ExerciseClassifier(assets);stream=RecognitionStream(classifier);builder=FeatureBuilder(.5)
    out.mkdir(parents=True,exist_ok=False)
    session=dict(schema=1,engine='motionlab-auto-ai-v1',classifier=dict(
        architecture='plain_tcn',checkpoint_sha256=classifier.metadata['checkpoint_sha256'],
        onnx_sha256=classifier.metadata['onnx_sha256'],classes=classifier.metadata['classes'],
        prediction_policy='45 observations at 15 Hz; stride 8 observations; two consecutive argmax votes; no calibrated confidence threshold'),
        video=str(video),metadata=meta,created=datetime.now().isoformat(timespec='seconds'),
        status='running',segments=[],timeline=None,results=[],crop=None,
        annotated_video=str(out/'annotated.mp4'),sample_rate_hz=15,
        segmentation='Causal neural exercise predictions only. No CSV labels or reference counts are read.',
        rep_policy='Existing rule-based counters start only after AI class confirmation. Warm-up, class transitions and tracking gaps can miss cycles. Curls count per-arm cycles with near-simultaneous bilateral cycles merged.',
        limits='Demonstration prototype for one person and a static camera. Closed six-class recognition; unsupported actions may be misclassified. Scores are uncalibrated. 2D angles and rep estimates are not form/safety assessments. Streaming stabilization and automatic counter routing were not measured by the held-out window-classification benchmark.')
    save_json(out/'session.json',session)
    factor=min(1,1280/max(meta['width'],meta['height']))
    width=max(2,round(meta['width']*factor)//2*2);height=max(2,round(meta['height']*factor)//2*2)
    reader=VideoReader(video);pose=None;encoder=None
    finished_counts={k:0 for k in EXERCISES};active=None;all_reps=[];predictions=[]
    samples=0;usable_samples=0;decoded=0;last_preview=0.;status='complete';failure=None
    decision=dict(raw_class=None,stable_class=None,raw_score=None,status='warming_up',window_coverage=0.)
    poses=[];angles=angles_for([],width,height);last_source_frame=-1

    def finish_segment(end_frame):
        nonlocal active
        if active is None:return
        segment=active;active=None;rows=segment['rows']
        if not rows or (end_frame-segment['first_frame'])/meta['fps']<.5:return
        number=len(session['results'])+1;exercise=segment['exercise'];folder=out/f'{number:02d}_{exercise}'
        folder.mkdir(exist_ok=True)
        reps=per_rep_analysis(segment['analyzer'].events,rows,exercise)
        count=rows[-1]['gated_reps'];finished_counts[exercise]+=count
        timing=consistency([r['duration_s'] for r in reps]);ranges=consistency([r['range_deg'] for r in reps])
        segment_record=dict(exercise=exercise,first_frame=segment['first_frame'],end_frame=end_frame,
            side='both',curl_policy='per_arm',reference=None,label_source='neural_classifier')
        coverage=sum(r['pose_observation_usable'] for r in rows)/len(rows)
        feedback=['Exercise was predicted by the neural classifier; no CSV labels were supplied.',
            'Rep estimates start after AI confirmation. Inspect transitions and incomplete cycles before interpreting the count.']
        result=dict(segment=number,exercise=exercise,title=EXERCISES[exercise],count=count,reference=None,
            reference_error=None,curl_policy='per_arm',arm_selection='both',directory=str(folder),
            source_first_frame=segment['first_frame'],source_end_frame=end_frame,
            start_s=segment['first_frame']/meta['fps'],end_s=end_frame/meta['fps'],
            duration=timing,range=ranges,tracking_coverage=coverage,feedback=feedback,reps=reps,
            render_width=width,render_height=height,crop=None,inference_s=None,inference_fps=None,
            mode_counts={mode:rows[-1][mode+'_reps'] for mode in ('raw','smoothed','gated')})
        write_csv(folder/'measurements.csv',rows);write_csv(folder/'reps.csv',reps)
        save_json(folder/'summary.json',result);save_json(folder/'events.json',segment['analyzer'].events)
        make_plots(folder,rows,reps,exercise)
        session['segments'].append(segment_record);session['results'].append(result)
        all_reps.extend(dict(segment=number,**rep) for rep in reps)
        write_csv(out/'all_reps.csv',all_reps);save_json(out/'session.json',session)

    try:
        pose=PoseEstimator(assets/'pose_landmarker_lite.task')
        encoder=writer(out/'annotated.mp4',meta['fps'],width,height)
        with (out/'ai_measurements.csv').open('w',newline='',encoding='utf-8-sig') as measurements, \
                (out/'landmarks.jsonl').open('w',encoding='utf-8') as landmarks:
            table=None
            for index,timestamp,image in reader.frames(0,meta['frames']):
                if cancelled and cancelled():raise Cancelled()
                image=cv2.resize(image,(width,height));last_source_frame=index;decoded+=1
                if valid_sample(index,0,meta['fps'],15):
                    poses=pose.detect(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),round(timestamp*1000))
                    points=np.full((33,5),np.nan,np.float32)
                    if poses:points[:]=[[p[k] for k in ('x','y','z','visibility','presence')] for p in poses[0]]
                    features,usable=builder.update(points,timestamp,width,height)
                    decision=stream.update(features,usable,timestamp);samples+=1;usable_samples+=int(usable)
                    angles=angles_for(poses,width,height)
                    stable=decision['stable_class']
                    if active is not None and active['exercise']!=stable:finish_segment(index)
                    if active is None and stable in EXERCISES:
                        active=dict(exercise=stable,first_frame=index,rows=[],
                            analyzer=ExerciseAnalyzer(stable,timestamp,None,side='both',curl_policy='per_arm'))
                    current_counts=dict(finished_counts)
                    if active is not None:
                        eligible=usable and decision['status']=='ready' and decision['raw_class']==stable
                        row=active['analyzer'].update(poses if eligible else [],index,timestamp,width,height)
                        row.update(in_score_window=True,pose_observation_usable=bool(usable),
                            counter_input_accepted=bool(eligible),ai_status=decision['status'])
                        active['rows'].append(row)
                        current_counts[stable]+=row['gated_reps']
                    record=dict(frame=index,timestamp_s=timestamp,raw_class=decision['raw_class'],
                        stable_class=stable,model_score_uncalibrated=decision['raw_score'],status=decision['status'],
                        pose_usable=bool(usable),window_pose_coverage=decision['window_coverage'],**angles,
                        **{key+'_rep_estimate':value for key,value in current_counts.items()})
                    if table is None:table=csv.DictWriter(measurements,fieldnames=list(record));table.writeheader()
                    table.writerow(record)
                    landmarks.write(json.dumps(dict(frame=index,timestamp_s=timestamp,poses=poses))+'\n')
                    if decision['updated']:
                        probabilities=stream.probabilities
                        predictions.append(dict(frame=index,timestamp_s=timestamp,**{k:v for k,v in decision.items() if k!='updated'},
                            **{'p_'+key:float(probabilities[i]) if probabilities is not None else None
                               for i,key in enumerate(classifier.metadata['classes'])}))
                current_counts=dict(finished_counts)
                if active is not None and active['rows']:
                    current_counts[active['exercise']]+=active['rows'][-1]['gated_reps']
                image=overlay(image,poses,decision,angles,current_counts,timestamp);encoder.write(image)
                if not (out/'preview.jpg').exists() and decision['status']=='ready':cv2.imwrite(str(out/'preview.jpg'),image)
                now=time.perf_counter()
                if preview and now-last_preview>=.1:
                    scale=min(640/width,360/height)
                    preview(cv2.cvtColor(cv2.resize(image,(max(2,round(width*scale)),max(2,round(height*scale)))),cv2.COLOR_BGR2RGB));last_preview=now
                if progress and index%30==0:
                    progress(round((index+1)/meta['frames']*100),
                        f'AI: {TITLES.get(decision["stable_class"],decision["status"].replace("_"," "))} · {timestamp:.1f}s')
        if decoded!=meta['frames']:raise ValueError('Video ended before its reported frame count.')
    except Cancelled:status='cancelled'
    except Exception as error:status='error';failure=error
    finally:
        reader.close()
        if encoder is not None:encoder.release()
        if pose is not None:pose.close()
    finish_segment(last_source_frame+1)
    write_csv(out/'predictions.csv',predictions)
    session.update(status=status,decoded_frames=decoded,pose_observations=samples,
        usable_pose_observations=usable_samples,tracking_coverage=usable_samples/max(1,samples),
        per_exercise_rep_estimates=finished_counts,
        pose_inference_s=pose.inference_s if pose else 0.,classifier_inference_s=classifier.inference_s,
        classifier_windows=classifier.calls)
    if failure is not None:session['error']=str(failure)
    if status=='complete':session['video_sha256']=sha256(video)
    timeline_plot(out,predictions)
    export_report(out,session)
    # Add the AI timeline to the generated practical HTML summary.
    report=(out/'report.html').read_text(encoding='utf-8')
    import base64
    encoded=base64.b64encode((out/'ai_timeline.png').read_bytes()).decode('ascii')
    report=report.replace('</html>',f'<h2>Automatic exercise timeline</h2><img src="data:image/png;base64,{encoded}">'
        '<p>Model scores are uncalibrated. Warm-up and transitions can omit early repetitions.</p></html>')
    (out/'report.html').write_text(report,encoding='utf-8')
    session['end_to_end_s']=time.perf_counter()-started
    session['source_frames_per_processing_second']=decoded/max(session['end_to_end_s'],1e-9)
    save_json(out/'session.json',session)
    if status=='cancelled':raise Cancelled()
    if failure is not None:raise failure
    if progress:progress(100,'Automatic AI analysis complete')
    return out/'session.json'
