import cv2
import numpy as np

EDGES = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),
         (23,24),(23,25),(25,27),(24,26),(26,28),(27,29),(29,31),(28,30),(30,32)]


def overlay(frame, poses, row):
    height, width = frame.shape[:2]
    if poses:
        pose = poses[0]
        for a, b in EDGES:
            if all(pose[i]['visibility'] >= 0.5 and pose[i]['presence'] >= 0.5 for i in (a,b)):
                points = [(int(pose[i]['x']*width), int(pose[i]['y']*height)) for i in (a,b)]
                cv2.line(frame, *points, (0,220,100), max(1,round(width/640)))
    count=row.get('motion_gated_reps',row['gated_reps'])
    support=row.get('motion_support',row['side']).upper()
    context='' if row.get('in_score_window',True) else ' | CONTEXT'
    lines = [f"SQUATS | motion reps: {count} | support: {support} | {row['timestamp_s']:.2f} s{context}"]
    for side in ('left','right'):
        raw, gated = row[side+'_raw_angle'], row[side+'_gated_angle']
        raw_label = '--' if raw is None else f'{raw:.1f}'
        label = '--' if gated is None else f'{gated:.1f}'
        status = 'OK' if row[side+'_reliable'] else 'REJECT: '+row[side+'_rejection_reasons']
        lines.append(f"{side.upper()} knee: {label} deg | raw: {raw_label} | {status}")
    phase=row.get('motion_phase','--')
    reason=row.get('motion_rejection_reasons','') or 'valid body chain'
    lines.append(f'Phase: {phase} | {reason} | angles are 2D knee measurements')
    unit = width/1280
    scale = max(0.35,0.76*unit)
    step, margin = max(20,round(33*unit)), max(6,round(14*unit))
    panel_height = min(height,step*len(lines)+margin)
    cv2.rectangle(frame,(0,0),(width,panel_height),(25,25,25),-1)
    for i, text in enumerate(lines):
        text_width = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,scale,1)[0][0]
        fit = min(scale,scale*(width-2*margin)/max(text_width,1))
        color = (255,255,255) if i in (0,3) else (100,235,255) if i==1 else (255,220,130)
        cv2.putText(frame,text,(margin,step*(i+1)),cv2.FONT_HERSHEY_SIMPLEX,fit,color,max(1,round(unit)),cv2.LINE_AA)
    return frame


def plot_comparison(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, constrained_layout=True)
    times = [r['timestamp_s'] for r in rows]
    for mode in ('raw','smoothed','gated'):
        values = np.array([r[mode+'_angle'] for r in rows], float)
        axes[0].plot(times, values, label=mode, alpha=0.8)
        axes[1].plot(times, [r[mode+'_reps'] for r in rows], label=mode)
    axes[0].set(ylabel='2D knee angle (degrees)', title='Same landmark pass; gaps are missing/rejected frames')
    axes[1].set(xlabel='Time (seconds)', ylabel='Completed repetitions')
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_motion(rows,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True,constrained_layout=True)
    times=[r['timestamp_s'] for r in rows]
    for mode in ('raw','smoothed','gated'):
        axes[0].plot(times,np.array([r['motion_'+mode+'_value'] for r in rows],float),label=mode)
        axes[1].plot(times,[r['motion_'+mode+'_reps'] for r in rows],label=mode)
    valid=[r['timestamp_s'] for r in rows if r.get('in_score_window',True)]
    if valid:
        for axis in axes:
            axis.axvspan(valid[0],valid[-1],color='green',alpha=.08,label='scoring window')
    axes[0].set(ylabel='Normalized hip descent',title='Body-motion counter; context confirms observed complete cycles')
    axes[1].set(xlabel='Source time (seconds)',ylabel='Assigned repetitions')
    for axis in axes:
        axis.legend();axis.grid(alpha=.2)
    fig.savefig(path,dpi=150);plt.close(fig)


def plot_bilateral(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,1,figsize=(11,6),sharex=True,constrained_layout=True)
    times = [r['timestamp_s'] for r in rows]
    for side in ('left','right'):
        axes[0].plot(times,np.array([r[side+'_gated_angle'] for r in rows],float),label=side+' knee (gated EMA)')
        axes[1].plot(times,[r[side+'_gated_reps'] for r in rows],label=side+' counter')
    axes[0].set(ylabel='2D knee angle (degrees)',title='Bilateral diagnostics; do not sum left/right repetition counts')
    axes[1].set(xlabel='Source time (seconds)',ylabel='Completed repetitions')
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.2)
    fig.savefig(path,dpi=150)
    plt.close(fig)
