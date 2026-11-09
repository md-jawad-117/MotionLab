"""PyQt6 desktop workspace. Run with python -m motionlab.gui."""
import argparse,csv,html,json,os,shutil,sys,traceback,uuid
from pathlib import Path
from datetime import datetime
os.environ.setdefault('QT_API','pyqt6')
os.environ.setdefault('OPENCV_FFMPEG_THREADS','1')
import cv2
import numpy as np
from PyQt6.QtCore import Qt,QThread,pyqtSignal,QTimer,QUrl
from PyQt6.QtGui import QImage,QPixmap,QDesktopServices,QFont,QFontDatabase
from PyQt6.QtWidgets import (QApplication,QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,
    QLabel,QPushButton,QFileDialog,QComboBox,QTableWidget,QTableWidgetItem,
    QHeaderView,QAbstractItemView,QFrame,QTabWidget,QTextBrowser,QProgressBar,QSlider,
    QMessageBox,QSizePolicy,QMenu)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from .exercises import EXERCISES
from .workout import probe_video,import_labels,validate_segments,run_workout,export_report,Cancelled

PROJECT=Path(__file__).resolve().parents[2]

def clock(seconds):
    seconds=max(0,float(seconds));minutes=int(seconds//60)
    return f'{minutes:02d}:{seconds-minutes*60:06.3f}'

def number(value,suffix='',digits=2):
    return '—' if value is None else f'{value:.{digits}f}{suffix}'

def button(text,slot,primary=False):
    widget=QPushButton(text);widget.clicked.connect(slot)
    if primary:widget.setObjectName('primary')
    return widget

class VideoPanel(QWidget):
    def __init__(self):
        super().__init__();self.capture=None;self.fps=30;self.frames=0;self.offset=0;self.current=0;self.image=None
        layout=QVBoxLayout(self);layout.setContentsMargins(0,0,0,0)
        self.screen=QLabel('Open a video to begin\n\nSource and annotated playback · silent preview')
        self.screen.setAlignment(Qt.AlignmentFlag.AlignCenter);self.screen.setMinimumHeight(240)
        self.screen.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding)
        self.screen.setObjectName('video');layout.addWidget(self.screen,1)
        row=QHBoxLayout();self.play=button('▶ Play',self.toggle);self.play.setEnabled(False)
        self.slider=QSlider(Qt.Orientation.Horizontal);self.slider.sliderMoved.connect(self.seek)
        self.time=QLabel('00:00.000');self.time.setMinimumWidth(95)
        row.addWidget(self.play);row.addWidget(self.slider,1);row.addWidget(self.time);layout.addLayout(row)
        self.timer=QTimer(self);self.timer.timeout.connect(self.advance)

    def load(self,path,offset=0):
        self.stop()
        if self.capture:self.capture.release()
        self.capture=cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            self.capture=None;raise ValueError('Cannot open video preview: '+str(path))
        self.fps=self.capture.get(cv2.CAP_PROP_FPS) or 30
        self.frames=int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT));self.offset=offset
        self.slider.setRange(0,max(0,self.frames-1));self.play.setEnabled(True);self.seek(0)

    def set_image(self,image):
        if isinstance(image,np.ndarray):
            image=QImage(image.data,image.shape[1],image.shape[0],image.strides[0],QImage.Format.Format_RGB888).copy()
        self.image=QPixmap.fromImage(image)
        self.screen.setPixmap(self.image.scaled(self.screen.size(),Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self,event):
        super().resizeEvent(event)
        if self.image:self.screen.setPixmap(self.image.scaled(self.screen.size(),Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation))

    def seek(self,index):
        if not self.capture:return
        self.capture.set(cv2.CAP_PROP_POS_FRAMES,max(0,min(int(index),self.frames-1)))
        self.advance()

    def seek_source(self,seconds):
        self.stop();self.seek(round((seconds-self.offset)*self.fps))

    def advance(self):
        if not self.capture:return
        ok,frame=self.capture.read()
        if not ok:self.stop();return
        self.current=max(0,int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))-1)
        self.set_image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        self.slider.setValue(self.current)
        timestamp=self.offset+self.current/self.fps
        self.time.setText(clock(timestamp))

    def toggle(self):
        if self.timer.isActive():self.stop()
        elif self.capture:
            if self.current>=self.frames-1:self.seek(0)
            self.timer.start(max(10,round(1000/self.fps)));self.play.setText('Ⅱ Pause')

    def stop(self):
        self.timer.stop();self.play.setText('▶ Play')

    def shutdown(self):
        self.stop()
        if self.capture:self.capture.release();self.capture=None

class AnalysisWorker(QThread):
    progress=pyqtSignal(int,str);frame=pyqtSignal(QImage);succeeded=pyqtSignal(str);failed=pyqtSignal(str);aborted=pyqtSignal(str)
    def __init__(self,video,segments,output,parent=None,timeline=None,automatic=False):
        super().__init__(parent);self.video=video;self.segments=segments;self.output=output;self.timeline=timeline;self.automatic=automatic

    def run(self):
        def preview(image):
            self.frame.emit(QImage(image.data,image.shape[1],image.shape[0],image.strides[0],QImage.Format.Format_RGB888).copy())
        try:
            if self.automatic:
                from .ai_workout import run_auto
                result=run_auto(self.video,self.output,self.progress.emit,preview,self.isInterruptionRequested)
            else:
                result=run_workout(self.video,self.segments,self.output,self.progress.emit,preview,self.isInterruptionRequested,timeline=self.timeline)
            self.succeeded.emit(str(result))
        except Cancelled:self.aborted.emit(str(Path(self.output)/'session.json'))
        except Exception as error:
            log=PROJECT/'runs/gui_errors.log';log.parent.mkdir(exist_ok=True)
            with log.open('a',encoding='utf-8') as handle:handle.write(traceback.format_exc()+'\n')
            self.failed.emit(str(error))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__();self.setWindowTitle('MotionLab');self.resize(1000,820)
        self.video=None;self.meta=None;self.segments=[];self.session=None;self.session_file=None
        self.worker=None;self._closing=False;self.label_mode=False
        root=QWidget();self.setCentralWidget(root)
        layout=QVBoxLayout(root);layout.setContentsMargins(14,10,14,10);layout.setSpacing(8)
        header=QHBoxLayout();brand=QLabel('MotionLab');brand.setObjectName('brand');header.addWidget(brand)
        subtitle=QLabel('EXERCISE STUDIO');subtitle.setObjectName('eyebrow');header.addWidget(subtitle);header.addStretch()
        layout.addLayout(header)
        self.source_label=QLabel('Open a video, choose the exercise, then analyze.');self.source_label.setObjectName('muted');layout.addWidget(self.source_label)
        controls=QHBoxLayout();controls.addWidget(QLabel('1'))
        self.open_video_button=button('Open video',self.open_video,True);controls.addWidget(self.open_video_button)
        controls.addSpacing(18);controls.addWidget(QLabel('2  Exercise'))
        self.quick_exercise=QComboBox()
        self.quick_exercise.addItem('Automatic (AI)','auto_ai')
        for key,title in EXERCISES.items():self.quick_exercise.addItem(title,key)
        self.quick_exercise.currentIndexChanged.connect(self.choose_quick_exercise);controls.addWidget(self.quick_exercise,1)
        controls.addSpacing(10);controls.addWidget(QLabel('3'))
        self.analyze_button=button('Analyze',self.analyze,True);self.analyze_button.setMinimumHeight(36);controls.addWidget(self.analyze_button)
        layout.addLayout(controls)
        self.player=VideoPanel();layout.addWidget(self.player,5)
        self.session_caption=QLabel('Your results will appear below the video.');self.session_caption.setObjectName('muted');layout.addWidget(self.session_caption)
        statrow=QHBoxLayout();self.cards={}
        for key,title in [('reps','DETECTED REPS'),('duration','MEAN REP DURATION'),('cv','TIMING VARIATION'),('coverage','TRACKING COVERAGE')]:
            card=QFrame();card.setObjectName('card');cardlay=QVBoxLayout(card)
            small=QLabel(title);small.setObjectName('eyebrow');value=QLabel('—');value.setObjectName('stat')
            if key=='reps':self.rep_title=small
            cardlay.addWidget(small);cardlay.addWidget(value);statrow.addWidget(card);self.cards[key]=value
        self.cards['cv'].setToolTip('Variation of detected rep durations. Lower means more uniform timing; it is not a form score.')
        layout.addLayout(statrow)
        resultbar=QHBoxLayout();resultbar.addWidget(QLabel('Results'))
        self.result_combo=QComboBox();self.result_combo.currentIndexChanged.connect(self.show_result);resultbar.addWidget(self.result_combo,1)
        self.details_button=button('Show details',self.toggle_details);self.details_button.setCheckable(True);resultbar.addWidget(self.details_button)
        self.save_results_button=QPushButton('Save results');self.save_results_button.setEnabled(False)
        menu=QMenu(self.save_results_button)
        for title,slot in [('Open results folder',self.open_folder),('Save annotated video',self.export_video),('Export rep data (CSV)',self.export_csv),('Save report (HTML)',self.export_html)]:
            menu.addAction(title,slot)
        self.save_results_button.setMenu(menu);resultbar.addWidget(self.save_results_button);layout.addLayout(resultbar)
        self.tabs=QTabWidget();layout.addWidget(self.tabs,4)
        chart_widget=QWidget();chart_layout=QVBoxLayout(chart_widget);chart_layout.setContentsMargins(6,6,6,6)
        self.chart_mode=QComboBox();self.chart_mode.addItems(['Movement & joint angles','Rep duration & joint range'])
        self.chart_mode.currentIndexChanged.connect(self.draw_chart);chart_layout.addWidget(self.chart_mode)
        self.figure=Figure(figsize=(9,3.5),layout='constrained');self.canvas=FigureCanvasQTAgg(self.figure)
        chart_layout.addWidget(self.canvas,1);self.tabs.addTab(chart_widget,'Data charts')
        self.rep_table=QTableWidget(0,8);self.rep_table.setHorizontalHeaderLabels(['Rep','Side','Start','Duration (s)','Range (°)','Left (°)','Right (°)','Tracking'])
        self.rep_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.rep_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.rep_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.rep_table.cellDoubleClicked.connect(self.jump_rep);self.tabs.addTab(self.rep_table,'Individual reps')
        self.insights=QTextBrowser();self.insights.setOpenExternalLinks(False)
        self.insights.setHtml('<h3>Your workout data</h3><p>Analyze a segment to inspect its timing, range and tracking.</p>')
        self.tabs.addTab(self.insights,'Insights & notes');self.tabs.hide()
        bottom=QHBoxLayout();bottom.addStretch()
        self.progress=QProgressBar();self.progress.setRange(0,100);self.progress.setValue(0);self.progress.setMaximumWidth(280);self.progress.hide()
        self.cancel_button=button('Stop analysis',self.cancel);self.cancel_button.setEnabled(False);self.cancel_button.hide()
        bottom.addWidget(self.progress);bottom.addWidget(self.cancel_button);layout.addLayout(bottom)
        self.output_controls=(self.save_results_button,)
        self.edit_controls=[self.open_video_button,self.quick_exercise,self.analyze_button]
        self.setStyleSheet(STYLE)

    def toggle_details(self,checked):
        self.tabs.setVisible(checked);self.details_button.setText('Hide details' if checked else 'Show details')
        if checked:QTimer.singleShot(0,self.draw_chart)

    def choose_quick_exercise(self,index):
        self.selection_caption()

    def set_status(self,text):
        self.progress.setToolTip(text)

    def notify(self,text):QMessageBox.information(self,'MotionLab',text)
    def error(self,text):QMessageBox.warning(self,'MotionLab',text)

    def refresh_exercise_choices(self,preferred=None):
        previous=preferred or self.quick_exercise.currentData()
        keys=[key for key in EXERCISES if not self.label_mode or any(s['exercise']==key for s in self.segments)]
        self.quick_exercise.blockSignals(True);self.quick_exercise.clear()
        self.quick_exercise.addItem('Automatic (AI)','auto_ai')
        for key in keys:self.quick_exercise.addItem(EXERCISES[key],key)
        index=self.quick_exercise.findData(previous);self.quick_exercise.setCurrentIndex(index if index>=0 else 0)
        self.quick_exercise.blockSignals(False)
        if self.label_mode:self.selection_caption()

    def selection_caption(self):
        if self.quick_exercise.currentData()=='auto_ai':
            self.session_caption.setText('AI predicts the exercise from movement. No CSV labels are used in this mode.')
            return
        if not self.label_mode:return
        key=self.quick_exercise.currentData();selected=[s for s in self.segments if s['exercise']==key]
        title=EXERCISES.get(key,'Exercise')
        self.session_caption.setText(f'CSV labels: {title} · {len(selected)} sections. Only these sections will be analyzed.')

    def selected_segments(self):
        if self.label_mode:
            return validate_segments([s for s in self.segments if s['exercise']==self.quick_exercise.currentData()],self.meta)
        return validate_segments([dict(exercise=self.quick_exercise.currentData(),first_frame=0,
                                       end_frame=self.meta['frames'],side='both',curl_policy='per_arm',reference=None)],self.meta)

    def open_video(self,path=None):
        if isinstance(path,bool):path=None
        if not path:path,_=QFileDialog.getOpenFileName(self,'Choose exercise video',str(PROJECT/'video'),'Videos (*.mp4 *.mov *.avi *.mkv);;All files (*)')
        if not path:return
        try:
            meta=probe_video(path);self.video=str(Path(path).resolve());self.meta=meta
            self.segments=[];self.session=None;self.session_file=None;self.label_mode=False
            self.refresh_exercise_choices();self.player.load(path);self.reset_results()
            self.source_label.setText(f'{Path(path).name}  ·  {clock(meta["duration_s"])}  ·  {meta["width"]} × {meta["height"]}  ·  {meta["fps"]:g} FPS')
            self.progress.setValue(0);self.session_caption.setText('Choose the exercise shown in your video, then Analyze.')
            labels=Path(path).with_name(Path(path).stem.replace('_rgb','')+'_labels.csv')
            if labels.is_file():
                try:
                    items,skipped=import_labels(labels,meta['fps'],meta['frames'])
                    if items:
                        self.segments=validate_segments(items,meta);self.label_mode=True
                        self.refresh_exercise_choices()
                        self.set_status(f'CSV loaded. {len(skipped)} unsupported sets skipped.')
                except ValueError:
                    self.set_status('Matching CSV could not be read; choose the exercise manually.')
            self.selection_caption()
            return True
        except Exception as error:self.error(str(error));return False

    def reset_results(self):
        self.rep_title.setText('DETECTED REPS')
        self.result_combo.blockSignals(True);self.result_combo.clear();self.result_combo.blockSignals(False);self.rep_table.setRowCount(0)
        for label in self.cards.values():label.setText('—')
        self.figure.clear();self.canvas.draw_idle()
        self.insights.setHtml('<h3>Your workout data</h3><p>Analyze a segment to see its timing, range and tracking notes.</p>')
        for widget in self.output_controls:widget.setEnabled(False)

    def analyze(self):
        if not self.video or not Path(self.video).is_file():self.notify('Open an available source video first.');return
        try:
            automatic=self.quick_exercise.currentData()=='auto_ai'
            items=[] if automatic else self.selected_segments()
            self.player.stop();self.player.play.setEnabled(False);self.reset_results();self.session=None;self.session_file=None
            out=PROJECT/'runs/gui'/('workout_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:5])
            for widget in self.edit_controls:widget.setEnabled(False)
            self.cancel_button.show();self.cancel_button.setEnabled(True);self.progress.show();self.progress.setValue(0);self.set_status('Starting local pose analysis…')
            self.worker=AnalysisWorker(self.video,items,str(out),self,
                timeline=list(self.segments) if self.label_mode and not automatic else None,automatic=automatic)
            self.worker.progress.connect(self.on_progress);self.worker.frame.connect(self.player.set_image)
            self.worker.succeeded.connect(self.load_session);self.worker.failed.connect(self.analysis_failed);self.worker.aborted.connect(self.analysis_cancelled)
            self.worker.finished.connect(self.worker_finished);self.worker.start()
        except Exception as error:self.error(str(error))

    def on_progress(self,value,text):self.progress.setValue(value);self.set_status(text)
    def cancel(self):
        if self.worker:self.worker.requestInterruption();self.cancel_button.setEnabled(False);self.set_status('Stopping safely after the current frame…')
    def analysis_failed(self,text):self.set_status('Analysis stopped: '+text);self.error(text+'\nCompleted outputs, if any, remain in the results folder.')
    def analysis_cancelled(self,path):
        if Path(path).exists():self.load_session(path)
        self.set_status('Analysis cancelled. Completed segments were preserved.')
    def worker_finished(self):
        for widget in self.edit_controls:widget.setEnabled(True)
        self.cancel_button.setEnabled(False);self.cancel_button.hide();self.progress.hide();self.player.play.setEnabled(self.player.capture is not None)
        worker=self.worker;self.worker=None
        if worker:worker.deleteLater()
        if self._closing:self.close()

    def load_session(self,path):
        try:
            session=json.loads(Path(path).read_text(encoding='utf-8'))
            if 'results' not in session or 'metadata' not in session:raise ValueError('Choose a MotionLab GUI session.json, not a CLI metrics file.')
            self.reset_results();self.session=session;self.session_file=Path(path)
            self.video=session['video'];self.meta=session['metadata'];self.segments=session.get('timeline') or session['segments']
            automatic=bool(session.get('classifier'))
            self.rep_title.setText('REP ESTIMATES' if automatic else 'DETECTED REPS')
            self.label_mode=bool(self.segments) and not automatic
            self.refresh_exercise_choices('auto_ai' if automatic else session['segments'][0]['exercise'] if session['segments'] else None)
            self.chart_mode.setItemText(0,'AI exercise timeline' if automatic else 'Movement & joint angles')
            self.source_label.setText(f'{Path(self.video).name} · {clock(self.meta["duration_s"])} · Saved session')
            results=session['results'];self.result_combo.blockSignals(True);self.result_combo.clear()
            for result in results:self.result_combo.addItem(f'{result["segment"]}. {result["title"]} · {result["count"]} reps · {clock(result["start_s"])}')
            self.result_combo.blockSignals(False)
            total=sum(r['count'] for r in results)
            self.session_caption.setText(f'{len(results)} AI exercise sections · {total} rep estimates · {session["status"]}' if automatic else
                f'{len(results)} analyzed segments · {total} detected repetitions · {session["status"]}')
            self.progress.setValue(100 if session['status']=='complete' else 0)
            self.save_results_button.setEnabled(True)
            if results:self.result_combo.setCurrentIndex(0);self.show_result(0)
            elif automatic:
                self.player.load(session['annotated_video'] if session.get('decoded_frames',0)>0 else self.video)
                self.insights.setHtml('<h3>Automatic exercise detection</h3><p>No supported exercise section was confirmed. '
                    'Review the full annotated video and AI timeline. Rest/setup and other exercise are recognized classes, '
                    'but do not activate repetition counters.</p><p>'+html.escape(session['limits'])+'</p>')
            self.set_status('Results saved locally.')
        except Exception as error:self.error(str(error))

    def active_result(self):
        index=self.result_combo.currentIndex()
        return self.session['results'][index] if self.session and 0<=index<len(self.session['results']) else None

    def show_result(self,index=0):
        result=self.active_result()
        if not result:return
        self.cards['reps'].setText(str(result['count']));self.cards['duration'].setText(number(result['duration']['mean'],' s'))
        self.cards['cv'].setText(number(result['duration']['cv_percent'],'%',1));self.cards['coverage'].setText(f'{result["tracking_coverage"]:.0%}')
        reps=result['reps'];self.rep_table.setRowCount(len(reps))
        for i,rep in enumerate(reps):
            values=[str(rep['rep']),rep['side'],clock(rep['start_s']),number(rep['duration_s']),number(rep['range_deg']),number(rep['left_range_deg']),number(rep['right_range_deg']),f'{rep["tracking_coverage"]:.0%}']
            for j,value in enumerate(values):self.rep_table.setItem(i,j,QTableWidgetItem(value))
        notes=''.join('<li>'+html.escape(text)+'</li>' for text in result['feedback']) or '<li>Inspect individual reps and the movement traces alongside the video.</li>'
        reference='No independent reference count supplied.' if result['reference'] is None else f'Detected {result["count"]}; supplied reference {result["reference"]}; absolute difference {result["reference_error"]}.'
        context=('Counters begin after AI confirmation; the first cycles can be missed.' if self.session.get('classifier') else
            'Each exercise segment is analyzed independently. Context can confirm a return, but only cycles with a turning point inside the selected segment are assigned.')
        self.insights.setHtml(f'<h2>{html.escape(result["title"])}</h2><p>{reference}</p><ul>{notes}</ul><h3>What the numbers mean</h3><p>Duration is the detected movement onset to observed return. Range is observed 2D knee/elbow angle excursion. Timing and range CV describe variability within this segment, not correct or safe form.</p><p>{html.escape(self.session["rep_policy"])}</p><p>{html.escape(self.session["limits"])}</p><p>{context}</p>')
        if self.session.get('classifier'):
            self.player.load(self.session['annotated_video']);self.player.seek_source(result['start_s'])
        else:self.player.load(Path(result['directory'])/'annotated.mp4',result['source_first_frame']/self.meta['fps'])
        self.draw_chart()

    def draw_chart(self,index=0):
        result=self.active_result()
        automatic=bool(self.session and self.session.get('classifier'))
        if not self.tabs.isVisible() or (not result and not automatic):return
        self.figure.clear();axes=self.figure.subplots(2,1)
        if automatic and self.chart_mode.currentIndex()==0:
            from .recognition import TITLES
            from .ml_data import CLASSES
            with (self.session_file.parent/'predictions.csv').open(encoding='utf-8-sig') as handle:rows=list(csv.DictReader(handle))
            times=[float(r['timestamp_s']) for r in rows]
            axes[0].step(times,[CLASSES.index(r['stable_class']) if r['stable_class'] in CLASSES else np.nan for r in rows],where='post',label='Confirmed AI class')
            axes[0].set(yticks=range(6),yticklabels=[TITLES[k] for k in CLASSES])
            axes[1].plot(times,[float(r['raw_score']) if r['raw_score'] else np.nan for r in rows],label='Uncalibrated model score')
            axes[1].plot(times,[float(r['window_coverage']) for r in rows],label='Usable pose coverage')
            axes[1].set(xlabel='Source time (s)',ylim=(0,1.05))
        elif not result:
            return
        elif self.chart_mode.currentIndex()==0:
            with (Path(result['directory'])/'measurements.csv').open(encoding='utf-8-sig') as handle:rows=list(csv.DictReader(handle))
            times=[float(r['timestamp_s']) for r in rows]
            for mode in ('raw','smoothed','gated'):
                values=[float(r[mode+'_signal']) if r[mode+'_signal'] else np.nan for r in rows]
                axes[0].plot(times,values,label=mode,linewidth=1.2)
            joint='elbow' if result['exercise'] in ('pushups','bicep_curls') else 'knee'
            for side in ('left','right'):
                axes[1].plot(times,[float(r[side+'_'+joint+'_deg']) if r[side+'_'+joint+'_deg'] else np.nan for r in rows],label=side+' '+joint,linewidth=1.2)
            axes[0].set_ylabel('Movement signal');axes[1].set_ylabel('2D angle (°)');axes[1].set_xlabel('Source time (s)')
            for axis in axes:axis.axvspan(result['start_s'],result['end_s'],color='#0c9588',alpha=.07)
        else:
            reps=result['reps'];indices=[r['rep'] for r in reps]
            axes[0].bar(indices,[r['duration_s'] for r in reps],color='#0c9588',label='Detected duration')
            axes[1].bar(indices,[r['range_deg'] if r['range_deg'] is not None else np.nan for r in reps],color='#557cc0',label='Observed joint range')
            axes[0].set_ylabel('Duration (s)');axes[1].set_ylabel('Range (°)');axes[1].set_xlabel('Repetition')
        for axis in axes:axis.grid(alpha=.18);axis.legend(fontsize=8,loc='upper right');axis.tick_params(labelsize=8)
        self.canvas.draw_idle()

    def jump_rep(self,row,column):
        result=self.active_result()
        if result and row<len(result['reps']):self.player.seek_source(result['reps'][row]['start_s'])
    def open_folder(self):
        if self.session_file:QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.session_file.parent)))
    def export_csv(self):
        if not self.session_file:return
        source=self.session_file.parent/'all_reps.csv'
        if not source.exists():self.notify('No completed repetitions to export yet.');return
        path,_=QFileDialog.getSaveFileName(self,'Export all detected repetitions',str(self.session_file.parent/'exported_reps.csv'),'CSV (*.csv)')
        if path:
            try:
                if Path(path).resolve()!=source.resolve():shutil.copy2(source,path)
                self.set_status('Per-repetition CSV exported.')
            except OSError as error:self.error(str(error))
    def export_video(self):
        result=self.active_result()
        automatic=bool(self.session and self.session.get('classifier'))
        if not result and not automatic:return
        if automatic and not self.session.get('decoded_frames',0):self.notify('No annotated frames were produced yet.');return
        suggested=self.session_file.parent/'exported_ai_video.mp4' if automatic else Path(result['directory'])/'exported_video.mp4'
        path,_=QFileDialog.getSaveFileName(self,'Save annotated video',str(suggested),'MP4 (*.mp4)')
        if path:
            try:
                source=Path(self.session['annotated_video']) if self.session.get('classifier') else Path(result['directory'])/'annotated.mp4'
                if Path(path).resolve()!=source.resolve():shutil.copy2(source,path)
                self.set_status('Annotated segment video saved.')
            except OSError as error:self.error(str(error))
    def export_html(self):
        if not self.session_file:return
        path,_=QFileDialog.getSaveFileName(self,'Save standalone analysis report',str(self.session_file.parent/'exported_report.html'),'HTML (*.html)')
        if path:
            try:export_report(Path(path).parent,self.session,Path(path).name);self.set_status('Report saved with embedded charts and rep tables.')
            except OSError as error:self.error(str(error))
    def closeEvent(self,event):
        if self.worker and self.worker.isRunning():
            self._closing=True;self.cancel();event.ignore()
        else:self.player.shutdown();event.accept()

STYLE='''
QMainWindow,QWidget{font-family:"DejaVu Sans";font-size:12px;color:#203348;background:#f2f5f9}
QLabel#brand{font-size:29px;font-weight:750;color:#087f79} QLabel#eyebrow{font-size:10px;font-weight:700;color:#718095;letter-spacing:1px}
QLabel#muted{color:#65758a;font-size:11px} 
QFrame#card{background:white;border:1px solid #dce4ed;border-radius:10px}
QFrame#card QLabel{background:transparent} QLabel#stat{font-size:25px;font-weight:650;color:#173b48}
QLabel#video{background:#111e2c;color:#a8bacd;border-radius:10px;font-size:15px}
QPushButton{background:white;border:1px solid #d4dee9;border-radius:6px;padding:8px 11px;font-weight:600}
QPushButton:hover{border-color:#0b9186;background:#edf8f5} QPushButton:disabled{color:#a0adba;background:#eef2f6;border-color:#e0e6ee}
QPushButton#primary{background:#087f79;color:white;border-color:#087f79} QPushButton#primary:hover{background:#066d68}
QComboBox{background:white;border:1px solid #d8e1eb;border-radius:5px;padding:6px}
QTableWidget{background:white;alternate-background-color:#f3f7fb;border:1px solid #dce4ed;border-radius:5px;gridline-color:#edf1f5;selection-background-color:#d9f0eb;selection-color:#173c43}
QHeaderView::section{background:#edf3f7;padding:7px;border:0;border-bottom:1px solid #dce4ed;font-weight:600}
QTabWidget::pane{background:white;border:1px solid #dce4ed;border-radius:6px} QTabBar::tab{padding:8px 18px;background:#e7edf4;border:0;margin-right:3px}
QTabBar::tab:selected{background:white;color:#087f79;font-weight:700} QTextBrowser{background:white;border:0;padding:8px}
QProgressBar{border:1px solid #dce4ed;border-radius:5px;background:white;text-align:center;height:18px} QProgressBar::chunk{background:#44b5a8;border-radius:4px}
QSlider::groove:horizontal{height:5px;background:#d6e1ec;border-radius:2px} QSlider::handle:horizontal{background:#087f79;width:12px;margin:-4px 0;border-radius:6px}
'''

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--video');parser.add_argument('--session');parser.add_argument('--snapshot')
    args=parser.parse_args()
    app=QApplication(sys.argv[:1]);app.setApplicationName('MotionLab');app.setStyle('Fusion')
    # Use Matplotlib's existing open font for reliable native/headless rendering.
    import matplotlib
    fonts=Path(matplotlib.get_data_path())/'fonts/ttf'
    for filename in ('DejaVuSans.ttf','DejaVuSans-Bold.ttf'):
        QFontDatabase.addApplicationFont(str(fonts/filename))
    app.setFont(QFont('DejaVu Sans',10))
    window=MainWindow();window.show()
    if args.video:window.open_video(args.video)
    if args.session:window.load_session(args.session)
    if args.snapshot:
        def capture():
            Path(args.snapshot).parent.mkdir(parents=True,exist_ok=True);window.grab().save(args.snapshot);window.close()
        QTimer.singleShot(1600,capture)
    sys.exit(app.exec())

if __name__=='__main__':main()
