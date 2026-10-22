/// <reference types="vite/client" />
import { ChangeEvent, DragEvent, useEffect, useRef, useState } from "react";
import { Activity, AudioLines, BarChart3, CheckCircle2, CircleAlert, Clock3, FileAudio, FileUp, Fingerprint, Info, LockKeyhole, Mic, MicOff, RotateCcw, ShieldCheck, Sparkles, UploadCloud, X } from "lucide-react";

type Prediction = { status: string; label: string | null; scores: Record<string, number> | null; model_version: string | null; note: string };
type Result = {
  analysis_id: string; status: string; filename: string;
  audio_quality: { status: string; warnings: string[] };
  vocal_presentation: Prediction; age_group: Prediction;
  acoustic_features: Record<string, number | null>;
  model_info: { feature_pipeline: string; feature_pipeline_version: string; classification_model: string | null; mode: string; model_enabled: boolean; accuracy_established: boolean };
  limitations: string[];
};
type BatchRow = { filename: string; status: string; result?: Result; error?: unknown };
const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
const ACCEPT = ".wav,.flac,.ogg,.aiff,.aif,.mp3,.m4a,.webm";
function fmt(value: unknown, unit = "") {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toLocaleString(undefined, { maximumFractionDigits: 2 }) + unit : "—";
}

export default function App() {
  const [file,setFile] = useState<File|null>(null);
  const [url,setUrl] = useState("");
  const [report,setReport] = useState<Result|null>(null);
  const [error,setError] = useState("");
  const [busy,setBusy] = useState(false);
  const [recording,setRecording] = useState(false);
  const [tab,setTab] = useState<"single"|"batch">("single");
  const [batchFiles,setBatchFiles] = useState<File[]>([]);
  const [batchRows,setBatchRows] = useState<BatchRow[]>([]);
  const [batchBusy,setBatchBusy] = useState(false);
  const [limits,setLimits] = useState(false);
  const [consent,setConsent] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const multiInput = useRef<HTMLInputElement>(null);
  const recorder = useRef<MediaRecorder|null>(null);
  const chunks = useRef<BlobPart[]>([]);
  const stream = useRef<MediaStream|null>(null);

  useEffect(() => () => {
    if (url) URL.revokeObjectURL(url);
    stream.current?.getTracks().forEach(track => track.stop());
  }, [url]);

  function selectFile(next?: File) {
    if (!next) return;
    setError(""); setReport(null); setFile(next); setUrl(URL.createObjectURL(next));
  }
  function handleDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault(); selectFile(e.dataTransfer.files?.[0]);
  }
  async function startRecord() {
    if (!consent) { setError("Please review and accept the audio-processing consent before recording."); return; }
    try {
      if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) throw new Error("Microphone recording is not supported in this browser. Upload a file instead.");
      const inputStream = await navigator.mediaDevices.getUserMedia({audio:true});
      stream.current = inputStream; chunks.current = [];
      const mime = MediaRecorder.isTypeSupported("audio/webm;codecs=opus") ? "audio/webm;codecs=opus" : "";
      const nextRecorder = new MediaRecorder(inputStream, mime ? {mimeType:mime} : undefined);
      recorder.current = nextRecorder;
      nextRecorder.ondataavailable = e => { if(e.data.size) chunks.current.push(e.data); };
      nextRecorder.onstop = () => {
        const blob = new Blob(chunks.current,{type:nextRecorder.mimeType || "audio/webm"});
        selectFile(new File([blob],"microphone-recording.webm",{type:blob.type}));
        inputStream.getTracks().forEach(t=>t.stop()); stream.current=null; setRecording(false);
      };
      nextRecorder.start(250); setRecording(true); setError("");
    } catch(e) { setError(e instanceof Error ? e.message : "Microphone permission failed."); }
  }
  async function runAnalysis() {
    if(!consent) { setError("Consent is required before audio analysis."); return; }
    if(!file) return;
    setBusy(true); setError(""); setReport(null);
    try {
      const form = new FormData(); form.append("file",file); form.append("consent","true");
      const response = await fetch(API+"/api/v1/analyze",{method:"POST",body:form});
      const body = await response.json();
      if(!response.ok) throw new Error(body?.detail?.message || body?.detail || "Analysis failed.");
      setReport(body as Result);
    } catch(e) { setError((e instanceof Error ? e.message : "Unable to reach the API.")+" API: "+API); }
    finally { setBusy(false); }
  }
  async function runBatch() {
    if(!consent) { setError("Consent is required before batch audio analysis."); return; }
    if(!batchFiles.length) return;
    setBatchBusy(true); setError(""); setBatchRows([]);
    try {
      const form=new FormData(); batchFiles.forEach(f=>form.append("files",f)); form.append("consent","true");
      const response=await fetch(API+"/api/v1/analyze/batch",{method:"POST",body:form});
      const body=await response.json();
      if(!response.ok) throw new Error(body?.detail?.message || "Batch analysis failed.");
      setBatchRows(body.results as BatchRow[]);
    } catch(e) { setError(e instanceof Error ? e.message : "Unable to analyze batch."); }
    finally { setBatchBusy(false); }
  }
  function exportBatch(format:"csv"|"json") {
    const rows=batchRows.map(row=>({
      filename:row.filename,status:row.status,duration_seconds:row.result?.acoustic_features.duration_seconds ?? "",
      f0_median_hz:row.result?.acoustic_features.f0_median_hz ?? "",
      age_group:row.result?.age_group.label || "not available",
      vocal_presentation:row.result?.vocal_presentation.label || "not available",
      quality:row.result?.audio_quality.status || "",error:row.error ? JSON.stringify(row.error) : ""
    }));
    const keys=Object.keys(rows[0]||{filename:"",status:""});
    const content=format==="json" ? JSON.stringify(rows,null,2) :
      [keys.join(","),...rows.map(r=>keys.map(k=>'"'+String((r as any)[k]??"").replace(/"/g,'""')+'"').join(","))].join("\n");
    const blob=new Blob([content],{type:format==="json"?"application/json":"text/csv"});
    const href=URL.createObjectURL(blob); const a=document.createElement("a");
    a.href=href;a.download="vgrs-batch-results."+format;a.click();URL.revokeObjectURL(href);
  }
  const m=report?.acoustic_features;
  const cards: [string,number|null|undefined,string,string][] = [
    ["Fundamental frequency",m?.f0_median_hz," Hz","Median tracked pitch; not a demographic label."],
    ["Pitch variation",m?.f0_std_hz," Hz","Standard deviation of tracked pitch."],
    ["Spectral centroid",m?.spectral_centroid_hz," Hz","A spectral-brightness measurement."],
    ["Active audio",m?.active_audio_fraction == null ? null : m.active_audio_fraction*100,"%","Frames above a simple energy threshold."],
    ["Clipping",m?.clipping_fraction == null ? null : m.clipping_fraction*100,"%","Samples near maximum digital amplitude."],
    ["RMS energy",m?.rms_energy,"","Average signal energy."]
  ];

  return <div className="shell">
    <header className="topbar">
      <a className="brand" href="#" aria-label="VGRS home"><span className="brand-icon"><AudioLines size={23}/></span><span><b>VGRS</b><small>VOICE ANALYSIS STUDIO</small></span></a>
      <div className="top-actions"><span className="local-tag"><i/>Local-first processing</span><button className="icon-btn top-info" onClick={()=>setLimits(v=>!v)} aria-label="Show limitations"><Info size={18}/></button></div>
    </header>
    <main className="page">
      <section className="hero">
        <div className="eyebrow"><span/>SPEECH SIGNAL INTELLIGENCE</div>
        <h1>Voice Analysis <em>Studio</em></h1>
        <p>Explore measurable acoustic characteristics while keeping signal features separate from model-based age and vocal-presentation estimates.</p>
        <div className="hero-points"><span><ShieldCheck size={15}/> No default audio retention</span><span><Activity size={15}/> Transparent measurements</span><span><Fingerprint size={15}/> Not identity verification</span></div>
      </section>

      {limits && <div className="limit-banner"><Info size={18}/><div><b>Scientific and use limitations</b><p>Voice alone cannot establish gender identity or verified chronological age. Accent, language, vocal training, health, recording equipment, noise and training-data representation may affect predictions. Do not use this tool for eligibility or high-impact decisions.</p></div><button className="icon-btn" onClick={()=>setLimits(false)} aria-label="Dismiss"><X size={17}/></button></div>}

      <nav className="tabs" aria-label="Analysis mode">
        <button className={tab==="single"?"active":""} onClick={()=>setTab("single")}><FileAudio size={16}/>Single recording</button>
        <button className={tab==="batch"?"active":""} onClick={()=>setTab("batch")}><BarChart3 size={16}/>Batch workspace <small>BETA</small></button>
      </nav>
      <label className="consent-row"><input type="checkbox" checked={consent} onChange={e=>setConsent(e.target.checked)}/><span><b>Consent before processing</b><small>I consent to temporary analysis of this recording by the configured VGRS API. The application does not intentionally retain audio or reuse it for model training.</small></span></label>

      {tab==="single" ? <section className="workspace">
        <article className="panel input-panel">
          <div className="panel-heading"><span className="step">01</span><div><h2>Input recording</h2><p>Upload an existing sample or record from your microphone.</p></div><span className="tag">WAV · FLAC · MP3*</span></div>
          <input ref={input} className="hidden" type="file" accept={ACCEPT} aria-label="Choose audio file" onChange={(e:ChangeEvent<HTMLInputElement>)=>{selectFile(e.target.files?.[0]);e.target.value="";}}/>
          <div className="dropzone" onDragOver={e=>e.preventDefault()} onDrop={handleDrop}>
            <span className="upload-icon"><UploadCloud size={24}/></span><h3>Drop your audio file here</h3><p>or browse recordings on your device</p>
            <button className="btn secondary" onClick={()=>input.current?.click()}><FileUp size={16}/>Browse files</button>
            <div className="file-rules">25 MB maximum <i/> 2–120 seconds <i/> Audio only</div>
          </div>
          <div className="divider-label"><span>OR RECORD DIRECTLY</span></div>
          <button className={recording?"record-btn recording":"record-btn"} onClick={()=>recording?recorder.current?.stop():startRecord()}>{recording?<><MicOff size={17}/>Stop recording <small>RECORDING</small></>:<><Mic size={17}/>Record with microphone</>}</button>
          {file && <div className="file-row"><span className="file-icon"><FileAudio size={19}/></span><div className="file-name"><b>{file.name}</b><small>{(file.size/1048576).toFixed(2)} MB · {file.type||"Audio file"}</small></div><button className="icon-btn" aria-label="Remove file" onClick={()=>{setFile(null);setUrl("");setReport(null);}}><X size={16}/></button></div>}
          {url && <audio className="player" controls src={url} aria-label="Audio preview"/>}
          <div className="upload-footer"><span><LockKeyhole size={14}/>Temporary processing; no intentional retention</span><button onClick={()=>{setFile(null);setUrl("");setReport(null);setError("");}}>Clear selection</button></div>
        </article>

        <article className="panel report-panel">
          <div className="panel-heading"><span className="step">02</span><div><h2>Analysis report</h2><p>Quality checks, signal measurements and model status.</p></div><span className="api-tag"><i/>On demand</span></div>
          {!report ? <div className="empty-report">
            <div className="signal-art"><span/><span/><span/><AudioLines size={28}/></div>
            <h3>{busy?"Analyzing recording…":"Your report starts here"}</h3>
            <p>{busy?"Validating the audio and extracting acoustic measurements.":"Choose a voice recording to create an acoustic report. Classification remains unavailable until a suitable real model is configured."}</p>
            {busy?<div className="progress"><span/></div>:<div className="check-list"><span><CheckCircle2 size={14}/>Input validation</span><span><CheckCircle2 size={14}/>Signal features</span><span><CheckCircle2 size={14}/>Model transparency</span></div>}
            <button className="btn primary analyze-btn" disabled={!file||busy||!consent} onClick={runAnalysis}><Sparkles size={17}/>{busy?"Analyzing…":"Analyze recording"}<b>→</b></button>
            <small className="footnote">Unavailable models never produce fabricated predictions.</small>
          </div>:<div className="result">
            <div className="result-ok"><CheckCircle2 size={19}/><div><b>Acoustic analysis complete</b><small>Analysis ID · {report.analysis_id.slice(0,8)}</small></div><button className="icon-btn" onClick={()=>setReport(null)} aria-label="Clear report"><RotateCcw size={16}/></button></div>
            <div className="pred-card"><span className="pred-icon"><Fingerprint size={19}/></span><div><small>VOCAL PRESENTATION</small><h3>{report.vocal_presentation.label||"Not enabled"}</h3><p>No validated classifier is configured for this build.</p></div><b className="status-chip">UNAVAILABLE</b></div>
            <div className="pred-card"><span className="pred-icon blue"><Clock3 size={19}/></span><div><small>ESTIMATED AGE GROUP</small><h3>{report.age_group.label||"Not enabled"}</h3><p>No validated classifier is configured for this build.</p></div><b className="status-chip">UNAVAILABLE</b></div>
            <div className="info-note"><Info size={16}/>This build provides actual acoustic features only; age and vocal-presentation predictions are intentionally withheld.</div>
            <button className="btn primary full" onClick={runAnalysis}><RotateCcw size={15}/>Run analysis again</button>
          </div>}
        </article>
      </section>:<section className="panel batch-panel">
        <div className="panel-heading"><span className="step">01</span><div><h2>Batch analysis</h2><p>Analyze several files with per-recording results and export.</p></div><span className="tag">UP TO 20 FILES</span></div>
        <input ref={multiInput} className="hidden" type="file" multiple accept={ACCEPT} aria-label="Choose multiple recordings" onChange={e=>setBatchFiles(Array.from(e.target.files||[]).slice(0,20))}/>
        <div className="batch-select"><span className="upload-icon"><FileUp size={24}/></span><div><b>Choose multiple audio recordings</b><p>Each file receives an independent validation status.</p></div><button className="btn secondary" onClick={()=>multiInput.current?.click()}>Select files</button></div>
        {batchFiles.length>0&&<div className="batch-file-list">{batchFiles.map((f,i)=><div className="batch-file" key={f.name+f.size}><FileAudio size={16}/><span>{f.name}</span><small>{(f.size/1048576).toFixed(2)} MB</small><button className="icon-btn" aria-label={"Remove "+f.name} onClick={()=>setBatchFiles(old=>old.filter((_,n)=>n!==i))}><X size={15}/></button></div>)}</div>}
        <div className="batch-actions"><button className="btn primary" disabled={!batchFiles.length||batchBusy||!consent} onClick={runBatch}><Sparkles size={16}/>{batchBusy?"Processing…":"Analyze "+batchFiles.length+" files"}</button><small>Temporary processing by local API; no training use.</small></div>
        {batchRows.length>0&&<><div className="batch-result-head"><h3>Results <span>{batchRows.length}</span></h3><div><button className="btn secondary compact" onClick={()=>exportBatch("csv")}>Export CSV</button><button className="btn secondary compact" onClick={()=>exportBatch("json")}>Export JSON</button></div></div><div className="table-wrap"><table><thead><tr><th>Recording</th><th>Status</th><th>Duration</th><th>Median F0</th><th>Quality</th><th>Predictions</th></tr></thead><tbody>{batchRows.map(row=><tr key={row.filename}><td>{row.filename}</td><td><span className={row.status==="completed"?"table-ok":"table-bad"}>{row.status}</span></td><td>{fmt(row.result?.acoustic_features.duration_seconds," s")}</td><td>{fmt(row.result?.acoustic_features.f0_median_hz," Hz")}</td><td>{row.result?.audio_quality.status||"—"}</td><td>Not enabled</td></tr>)}</tbody></table></div></>}
      </section>}

      {error&&<div className="error" role="alert"><CircleAlert size={17}/><span>{error}</span><button className="icon-btn" onClick={()=>setError("")} aria-label="Dismiss error"><X size={15}/></button></div>}

      {tab==="single"&&report&&<section className="details">
        <div className="section-title"><div className="eyebrow"><span/>MEASURED SIGNAL</div><h2>Acoustic feature detail</h2><small>Pipeline v{report.model_info.feature_pipeline_version}</small></div>
        {report.audio_quality.warnings.length>0&&<div className="warnings"><b><CircleAlert size={16}/>Recording quality notes</b>{report.audio_quality.warnings.map(w=><p key={w}>{w}</p>)}</div>}
        <div className="metrics">{cards.map(([name,value,unit,description])=><article className="metric" key={name}><div><span>{name}</span><span title={description}><Info size={14}/></span></div><strong>{fmt(value,unit)}</strong><small>{description}</small><div className="meter"><span style={{width:typeof value==="number"?Math.max(4,Math.min(100,Number(value)/(unit==="%"?1:unit===" Hz"?8:1)*100))+"%":"0%"}}/></div></article>)}</div>
        <div className="meta-panel"><b><Activity size={16}/>Processing metadata</b><div><span><small>Duration</small><strong>{fmt(m?.duration_seconds," seconds")}</strong></span><span><small>Output sample rate</small><strong>{fmt(m?.sample_rate_hz," Hz")}</strong></span><span><small>Input channels</small><strong>{fmt(m?.channel_count)}</strong></span><span><small>Classifier</small><strong>Disabled</strong></span></div></div>
        <div className="disclaimer"><ShieldCheck size={18}/><div><b>Interpret measurements in context</b><p>Signal features are not diagnoses or reliable identity claims. Jitter, shimmer, speaking rate and SNR are marked unavailable because this baseline does not measure them.</p></div></div>
      </section>}

      <section className="trust-row"><div><LockKeyhole size={18}/><span><b>Audio not retained by default</b><small>Temporary in-memory processing.</small></span></div><div><Activity size={18}/><span><b>Measured, not made up</b><small>Missing values remain unavailable.</small></span></div><div><ShieldCheck size={18}/><span><b>Research-first safeguards</b><small>No unsupported accuracy claims.</small></span></div></section>
    </main>
    <footer><span>VGRS <i>/</i> Voice Analysis Studio</span><span>Local research prototype · v0.1.0</span></footer>
  </div>;
}
