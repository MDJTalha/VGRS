/// <reference types="vite/client" />
import { useEffect, useMemo, useRef, useState, type ChangeEvent, type DragEvent, type ReactNode } from "react";
import {
  Activity, AlertCircle, ArrowRight, AudioLines, BarChart3, Bell, Check, CheckCircle2,
  ChevronDown, ChevronRight, CircleHelp, Clock3, FileAudio, FileClock, FileText, Fingerprint,
  Gauge, History, Home, Info, LayoutDashboard, LoaderCircle, LockKeyhole, Menu, Mic, MicOff,
  PanelLeftClose, PanelLeftOpen, Plus, RefreshCw, ScanSearch, Settings2, ShieldCheck,
  SlidersHorizontal, Sparkles, Upload, UploadCloud, X, Zap
} from "lucide-react";
import type { LucideIcon } from "lucide-react";

type Prediction = { status: string; label: string | null; scores: Record<string, number> | null; model_version: string | null; note: string };
type AnalysisResult = {
  analysis_id: string; status: string; filename: string;
  audio_quality: { status: string; warnings: string[] };
  vocal_presentation: Prediction; age_group: Prediction;
  acoustic_features: Record<string, number | null>;
  model_info: { feature_pipeline: string; feature_pipeline_version: string; classification_model: string | null; mode: string; model_enabled: boolean; accuracy_established: boolean; classification_status?: string; production_approved?: boolean };
  limitations: string[];
};
type ActivityItem = { id: string; filename: string; createdAt: string; result: AnalysisResult; source: "single" | "batch" };
type BatchRow = { filename: string; status: string; result?: AnalysisResult; error?: unknown };
type Page = "overview" | "analyze" | "batch" | "history" | "settings";
type Health = { status: "checking" | "online" | "offline"; version?: string; classificationReady?: boolean };
const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
const ACCEPT = ".wav,.flac,.ogg,.aiff,.aif,.mp3,.m4a,.webm";
const TITLES: Record<Page, string> = { overview: "Overview", analyze: "Voice analysis", batch: "Batch workspace", history: "Session history", settings: "System settings" };
const SUBTITLES: Record<Page, string> = {
  overview: "A clear view of your analysis workspace and service readiness.",
  analyze: "Inspect a recording and review measured acoustic features.",
  batch: "Validate multiple recordings and export per-file results.",
  history: "Review completed reports from this browser session.",
  settings: "Review service connectivity, model availability and safeguards."
};
function fmt(value: unknown, unit = "") {
  return typeof value === "number" && Number.isFinite(value) ? value.toLocaleString(undefined, { maximumFractionDigits: 2 }) + unit : "—";
}
function reportId(value: string) { return value.slice(0, 8).toUpperCase(); }
function timestamp(value: string) {
  try { return new Date(value).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }); }
  catch { return value; }
}

export default function App() {
  const [page, setPage] = useState<Page>("overview");
  const [collapsed, setCollapsed] = useState(false);
  const [mobileNav, setMobileNav] = useState(false);
  const [health, setHealth] = useState<Health>({ status: "checking" });
  const [models, setModels] = useState<any>(null);
  const [activities, setActivities] = useState<ActivityItem[]>([]);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState("");
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  const [consent, setConsent] = useState(false);
  const [recording, setRecording] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [showResearchNotes, setShowResearchNotes] = useState(false);
  const [batchFiles, setBatchFiles] = useState<File[]>([]);
  const [batchRows, setBatchRows] = useState<BatchRow[]>([]);
  const [batchBusy, setBatchBusy] = useState(false);
  const uploadRef = useRef<HTMLInputElement>(null);
  const batchRef = useRef<HTMLInputElement>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const streamRef = useRef<MediaStream | null>(null);

  useEffect(() => {
    let live = true;
    const refresh = async () => {
      try {
        const [healthResponse, modelsResponse] = await Promise.all([fetch(API + "/api/v1/health"), fetch(API + "/api/v1/models")]);
        if (!healthResponse.ok || !modelsResponse.ok) throw new Error("API check failed");
        const [healthBody, modelsBody] = await Promise.all([healthResponse.json(), modelsResponse.json()]);
        if (live) {
          setHealth({ status: "online", version: healthBody.version, classificationReady: Boolean(healthBody.classification_ready) });
          setModels(modelsBody);
        }
      } catch {
        if (live) setHealth({ status: "offline" });
      }
    };
    refresh();
    const timer = window.setInterval(refresh, 30000);
    return () => { live = false; window.clearInterval(timer); };
  }, []);

  useEffect(() => () => {
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    streamRef.current?.getTracks().forEach(track => track.stop());
  }, [audioUrl]);

  const recent = useMemo(() => activities.slice(0, 5), [activities]);
  const acousticModel = models?.models?.find((model: any) => model.kind === "acoustic_features");
  const classifierModel = models?.models?.find((model: any) => model.kind === "classification");

  function navigate(next: Page) {
    setPage(next); setMobileNav(false); setError(""); setNotice(""); setShowResearchNotes(false);
  }
  function chooseFile(file?: File) {
    if (!file) return;
    setSelectedFile(file); setAudioUrl(URL.createObjectURL(file)); setResult(null); setError(""); setNotice("");
  }
  function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    chooseFile(event.target.files?.[0]); event.target.value = "";
  }
  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault(); setDragging(false); chooseFile(event.dataTransfer.files?.[0]);
  }
  async function beginRecording() {
    if (!consent) { setError("Please confirm consent before recording or processing audio."); return; }
    try {
      if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) throw new Error("Microphone recording is not supported by this browser. Upload an audio file instead.");
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream; chunksRef.current = [];
      const mime = MediaRecorder.isTypeSupported("audio/webm;codecs=opus") ? "audio/webm;codecs=opus" : "";
      const recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
      recorderRef.current = recorder;
      recorder.ondataavailable = event => { if (event.data.size) chunksRef.current.push(event.data); };
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType || "audio/webm" });
        chooseFile(new File([blob], "microphone-recording.webm", { type: blob.type }));
        stream.getTracks().forEach(track => track.stop()); streamRef.current = null; setRecording(false);
      };
      recorder.start(250); setRecording(true); setError("");
    } catch (cause) { setError(cause instanceof Error ? cause.message : "Microphone access failed. Check browser permissions."); }
  }
  async function analyzeOne() {
    if (!consent) { setError("Consent is required before audio analysis."); return; }
    if (!selectedFile) { setError("Choose an audio recording first."); return; }
    setBusy(true); setError(""); setNotice(""); setResult(null);
    try {
      const form = new FormData(); form.append("file", selectedFile); form.append("consent", "true");
      const response = await fetch(API + "/api/v1/analyze", { method: "POST", body: form });
      const body = await response.json();
      if (!response.ok) throw new Error(body?.detail?.message || body?.detail || "The API could not analyze this recording.");
      const analysis = body as AnalysisResult;
      setResult(analysis);
      setActivities(old => [{ id: analysis.analysis_id, filename: selectedFile.name, createdAt: new Date().toISOString(), result: analysis, source: "single" }, ...old]);
      setNotice(analysis.model_info.production_approved
        ? "Analysis complete. Acoustic measurements and the configured classifier returned results under the recorded validation scope."
        : analysis.model_info.model_enabled
          ? "Analysis complete. Experimental model estimates are available, but they are not production-validated and must not be treated as verified age or gender identity."
          : "Analysis complete. Acoustic measurements are available; classification remains unavailable until a licensed, validated model is configured.");
    } catch (cause) {
      setError((cause instanceof Error ? cause.message : "Unable to reach the VGRS API.") + " Endpoint: " + API);
    } finally { setBusy(false); }
  }
  async function analyzeBatch() {
    if (!consent) { setError("Consent is required before batch analysis."); return; }
    if (!batchFiles.length) { setError("Select at least one recording for batch analysis."); return; }
    setBatchBusy(true); setError(""); setNotice(""); setBatchRows([]);
    try {
      const form = new FormData(); batchFiles.forEach(file => form.append("files", file)); form.append("consent", "true");
      const response = await fetch(API + "/api/v1/analyze/batch", { method: "POST", body: form });
      const body = await response.json();
      if (!response.ok) throw new Error(body?.detail?.message || "The batch request failed.");
      const rows = body.results as BatchRow[]; setBatchRows(rows);
      const completed: ActivityItem[] = rows.filter(row => row.result).map(row => ({
        id: row.result!.analysis_id, filename: row.filename, createdAt: new Date().toISOString(), result: row.result!, source: "batch"
      }));
      if (completed.length) setActivities(old => [...completed, ...old]);
      setNotice("Batch finished: " + body.completed + " completed, " + body.failed + " failed. Results retain per-file status.");
    } catch (cause) { setError(cause instanceof Error ? cause.message : "Unable to analyze the batch."); }
    finally { setBatchBusy(false); }
  }
  function exportBatch(format: "csv" | "json") {
    const rows = batchRows.map(row => ({
      filename: row.filename, status: row.status, analysis_id: row.result?.analysis_id || "",
      duration_seconds: row.result?.acoustic_features.duration_seconds ?? "",
      f0_median_hz: row.result?.acoustic_features.f0_median_hz ?? "",
      vocal_presentation: row.result?.vocal_presentation.label || "not available",
      age_group: row.result?.age_group.label || "not available", quality: row.result?.audio_quality.status || "",
      error: row.error ? JSON.stringify(row.error) : ""
    }));
    const keys = Object.keys(rows[0] || { filename: "", status: "" });
    const content = format === "json" ? JSON.stringify(rows, null, 2) :
      [keys.join(","), ...rows.map(row => keys.map(key => '"' + String((row as any)[key] ?? "").replace(/"/g, '""') + '"').join(","))].join("\n");
    const href = URL.createObjectURL(new Blob([content], { type: format === "json" ? "application/json" : "text/csv" }));
    const anchor = document.createElement("a"); anchor.href = href; anchor.download = "vgrs-batch-results." + format; anchor.click(); URL.revokeObjectURL(href);
  }
  function openActivity(item: ActivityItem) {
    setResult(item.result); setSelectedFile(null); setAudioUrl(""); setPage("analyze"); setMobileNav(false); setError(""); setNotice("Viewing a report retained in the current browser session.");
  }
  function clearSession() {
    setActivities([]); setResult(null); setNotice("Session activity cleared. Reports are not stored in a persistent history database.");
  }
  const featureCards: { name: string; key: string; value: number | null | undefined; unit: string; note: string; icon: LucideIcon }[] = [
    { name: "Median pitch (F0)", key: "f0_median_hz", value: result?.acoustic_features.f0_median_hz, unit: " Hz", note: "Median tracked fundamental frequency; not a demographic label.", icon: AudioLines },
    { name: "Pitch variation", key: "f0_std_hz", value: result?.acoustic_features.f0_std_hz, unit: " Hz", note: "Standard deviation across tracked voiced frames.", icon: Activity },
    { name: "Spectral centroid", key: "spectral_centroid_hz", value: result?.acoustic_features.spectral_centroid_hz, unit: " Hz", note: "Frequency-weighted spectral brightness indicator.", icon: Gauge },
    { name: "Active audio", key: "active_audio_fraction", value: result?.acoustic_features.active_audio_fraction == null ? null : result.acoustic_features.active_audio_fraction * 100, unit: "%", note: "Frames above a signal energy threshold.", icon: Zap },
    { name: "Clipping", key: "clipping_fraction", value: result?.acoustic_features.clipping_fraction == null ? null : result.acoustic_features.clipping_fraction * 100, unit: "%", note: "Samples close to maximum digital amplitude.", icon: SlidersHorizontal },
    { name: "RMS energy", key: "rms_energy", value: result?.acoustic_features.rms_energy, unit: "", note: "Average energy of the mono signal.", icon: BarChart3 }
  ];

  return <div className={"app-shell" + (collapsed ? " is-collapsed" : "")}>
    {mobileNav && <button className="mobile-scrim" aria-label="Close navigation" onClick={() => setMobileNav(false)} />}
    <aside className={"sidebar" + (mobileNav ? " mobile-open" : "")}>
      <div className="brand-row"><span className="brand-symbol"><AudioLines size={22}/></span>{!collapsed && <span className="brand-label"><b>VGRS</b><small>VOICE INTELLIGENCE</small></span>}<button className="collapse-control" title={collapsed ? "Expand sidebar" : "Collapse sidebar"} aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"} onClick={() => setCollapsed(value => !value)}>{collapsed ? <PanelLeftOpen size={16}/> : <PanelLeftClose size={16}/>}</button></div>
      {!collapsed && <div className="workspace-chip"><span className="workspace-glyph">L</span><span><b>Local workspace</b><small>Research environment</small></span><ChevronDown size={15}/></div>}
      {!collapsed && <div className="nav-caption">WORKSPACE</div>}
      <nav className="navigation" aria-label="Main navigation">
        <NavButton icon={LayoutDashboard} label="Overview" active={page === "overview"} collapsed={collapsed} onClick={() => navigate("overview")}/>
        <NavButton icon={ScanSearch} label="Voice analysis" active={page === "analyze"} collapsed={collapsed} onClick={() => navigate("analyze")}/>
        <NavButton icon={BarChart3} label="Batch workspace" active={page === "batch"} collapsed={collapsed} onClick={() => navigate("batch")}/>
        <NavButton icon={History} label="Session history" active={page === "history"} collapsed={collapsed} count={activities.length} onClick={() => navigate("history")}/>
      </nav>
      {!collapsed && <div className="nav-caption config-caption">CONFIGURATION</div>}
      <nav className="navigation"><NavButton icon={Settings2} label="System settings" active={page === "settings"} collapsed={collapsed} onClick={() => navigate("settings")}/><button className="nav-item" title="Help and limitations" onClick={() => setShowResearchNotes(v => !v)}><CircleHelp size={18}/>{!collapsed && <span>Help & limitations</span>}</button></nav>
      <div className="sidebar-spacer"/>
      <div className="sidebar-service"><span className={"service-led " + health.status}/>{!collapsed && <span><b>{health.status === "online" ? "API connected" : health.status === "checking" ? "Checking API" : "API unavailable"}</b><small>{health.status === "online" ? "Local service responding" : health.status === "checking" ? "Waiting for service" : "Check backend process"}</small></span>}</div>
      {!collapsed && <div className="sidebar-profile"><span className="profile-avatar">VR</span><span><b>VGRS Research</b><small>Local operator</small></span><span className="profile-dots">•••</span></div>}
    </aside>

    <div className="app-main">
      <header className="topbar"><button className="mobile-menu" aria-label="Open navigation" onClick={() => setMobileNav(true)}><Menu size={20}/></button>
        <div className="breadcrumb"><span>Workspace</span><ChevronRight size={14}/><b>{TITLES[page]}</b></div>
        <div className="top-actions"><div className={"api-status " + health.status}><i/>{health.status === "online" ? "Service online" : health.status === "checking" ? "Checking service" : "Service offline"}</div><button className="top-icon" aria-label="Toggle research notes" onClick={() => setShowResearchNotes(v => !v)}><Bell size={17}/></button><span className="top-divider"/><span className="top-profile">VR</span></div>
      </header>
      <main className="page-content">
        <div className="page-header"><div><div className="eyebrow">VGRS / RESEARCH ENVIRONMENT</div><h1>{page === "overview" ? "Voice Analysis Studio" : TITLES[page]}</h1><p>{SUBTITLES[page]}</p></div><div className="page-actions"><button className="button button-quiet" onClick={() => setShowResearchNotes(v => !v)}><Info size={16}/> Research notes</button><button className="button button-primary" onClick={() => { navigate("analyze"); setResult(null); setSelectedFile(null); setAudioUrl(""); }}><Plus size={16}/> New analysis</button></div></div>
        {showResearchNotes && <Callout kind="warning" icon={Info} title="Scientific and use limitations" onClose={() => setShowResearchNotes(false)}>Voice features cannot verify gender identity or chronological age. Classification is currently unavailable because no suitable validated model is enabled. Do not use voice-based estimates for eligibility, hiring, identity verification or other high-impact decisions.</Callout>}
        {error && <Callout kind="error" icon={AlertCircle} title="Action could not be completed" onClose={() => setError("")}>{error}</Callout>}
        {notice && <Callout kind="success" icon={CheckCircle2} title="Workspace update" onClose={() => setNotice("")}>{notice}</Callout>}

        {page === "overview" && <div className="overview">
          <section className="welcome">
            <div className="welcome-copy"><div className="welcome-eyebrow"><span><Sparkles size={13}/></span> SIGNAL PROCESSING WORKSPACE</div><h2>Understand the voice.<br/><em>Respect the evidence.</em></h2><p>Measure vocal acoustics, review recording quality, and keep model-based estimates grounded in validated evidence.</p><div className="welcome-buttons"><button className="button button-lime" onClick={() => navigate("analyze")}><Plus size={16}/> Start new analysis <ArrowRight size={16}/></button><button className="welcome-text-button" onClick={() => navigate("settings")}>Review system readiness <ArrowRight size={14}/></button></div></div>
            <div className="welcome-art" aria-hidden="true"><span className="art-orbit orbit-a"/><span className="art-orbit orbit-b"/><span className="art-orbit orbit-c"/><div className="art-wave">{[14,23,34,48,29,56,38,24,42,31,17].map((height,index)=><i key={index} style={{height}}/>)}</div><div className="art-core"><AudioLines size={34}/></div><div className="art-label"><span/>ACOUSTIC SIGNAL</div></div>
          </section>
          <section className="stats-grid">
            <StatCard label="Session analyses" value={String(activities.length)} description="Completed in this browser session" icon={FileClock} tint="green" footer="Not persisted to a database"/>
            <StatCard label="Acoustic pipeline" value={acousticModel?.status === "available" ? "Available" : health.status === "offline" ? "Offline" : "Checking"} description="Signal feature extraction" icon={Activity} tint="blue" footer={health.status === "online" ? "API is responding" : "Live service status"}/>
            <StatCard label="Classification models" value={classifierModel?.status === "available" ? "Ready" : classifierModel?.status === "experimental" ? "Research mode" : "Not enabled"} description="Age group & vocal presentation" icon={Fingerprint} tint="amber" footer={classifierModel?.production_approved ? "Validation report checked" : "No predictions are fabricated · no production validation approval"}/>
          </section>
          <div className="overview-columns">
            <section className="panel workflow-panel"><SectionHeading title="Analysis workspace" subtitle="Choose the workflow that fits your task." badge="WORKFLOWS"/>
              <WorkflowRow icon={ScanSearch} tone="violet" title="Single voice analysis" description="Inspect a recording with a detailed acoustic feature report." tags={["Pitch metrics","Quality checks"]} onClick={() => navigate("analyze")}/>
              <WorkflowRow icon={BarChart3} tone="blue" title="Batch analysis" description="Process multiple files and export per-file results." tags={["Up to 20 files","CSV / JSON"]} onClick={() => navigate("batch")}/>
              <WorkflowRow icon={History} tone="green" title="Session history" description="Return to a report created during this browser session." tags={[activities.length + " reports","Temporary"]} onClick={() => navigate("history")}/>
            </section>
            <section className="panel readiness-panel"><SectionHeading title="System readiness" subtitle="Live state from the configured API"/><Readiness icon={Activity} title="API service" detail={health.status === "online" ? "Connected and responding" : health.status === "checking" ? "Checking configured endpoint" : "Cannot reach the API"} value={health.status === "online" ? "Online" : health.status === "checking" ? "Checking" : "Offline"} tone={health.status === "online" ? "good" : health.status === "offline" ? "bad" : "neutral"}/><Readiness icon={AudioLines} title="Acoustic pipeline" detail={acousticModel?.status === "available" ? "Baseline measurements enabled" : "Waiting for model registry"} value={acousticModel?.status === "available" ? "Ready" : "Unknown"} tone={acousticModel?.status === "available" ? "good" : "neutral"}/><Readiness icon={Fingerprint} title="Demographic classifier" detail={classifierModel?.status === "experimental" ? "Loaded for research; VGRS validation is incomplete" : "Requires a licensed model and matching validation report"} value={classifierModel?.status === "available" ? "Validated" : classifierModel?.status === "experimental" ? "Experimental" : "Disabled"} tone={classifierModel?.status === "available" ? "good" : "warn"}/><Readiness icon={ShieldCheck} title="Consent gate" detail="Required in both the UI and API" value="Enabled" tone="good"/><div className="readiness-note"><Info size={14}/> Service availability is not a claim of model accuracy or production readiness.</div></section>
          </div>
          <section className="panel recent-panel"><SectionHeading title="Recent analyses" subtitle="Reports created in this browser session only." action={<button className="button button-quiet button-small" onClick={() => navigate("history")}>View session history <ArrowRight size={14}/></button>}/>
            {recent.length === 0 ? <div className="no-activity"><span className="no-activity-icon"><FileClock size={20}/></span><span><b>No activity yet</b><small>Completed reports will appear here. The system does not insert sample records or demo metrics.</small></span><button className="button button-outline button-small" onClick={() => navigate("analyze")}>Run first analysis <ArrowRight size={14}/></button></div> : <ActivityTable items={recent} onOpen={item => { setResult(item.result); setPage("analyze"); setSelectedFile(null); setAudioUrl(""); }}/>}
          </section>
        </div>}

        {page === "analyze" && <div className="analysis-page">
          <section className="analysis-columns">
            <div className="panel input-panel"><PanelHeading step="01" title="Audio input" subtitle="Upload a recording or record directly from your microphone." meta="SINGLE FILE"/>
              <input ref={uploadRef} className="visually-hidden" type="file" accept={ACCEPT} onChange={onFileChange} aria-label="Choose audio file"/>
              <div className={"upload-zone" + (dragging ? " is-dragging" : "")} onDragOver={event => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={onDrop}><div className="upload-mark"><UploadCloud size={25}/></div><h3>Drop audio file here</h3><p>or choose a recording from your computer</p><button className="button button-outline" onClick={() => uploadRef.current?.click()}><Upload size={16}/> Browse files</button><div className="constraints"><span>Maximum 25 MB</span><i/><span>2–120 seconds</span><i/><span>Audio only</span></div></div>
              <div className="or-divider"><span>OR CAPTURE AUDIO</span></div>
              <button className={"record-button" + (recording ? " recording" : "")} onClick={() => recording ? recorderRef.current?.stop() : beginRecording()}>{recording ? <MicOff size={17}/> : <Mic size={17}/>} {recording ? "Stop recording" : "Record with microphone"}{recording && <span className="live-badge"><i/> RECORDING</span>}{!recording && <small>Browser permission required</small>}</button>
              {selectedFile && <div className="selected-file"><span className="selected-file-symbol"><FileAudio size={19}/></span><span className="selected-file-name"><b>{selectedFile.name}</b><small>{(selectedFile.size / 1048576).toFixed(2)} MB · {selectedFile.type || "Audio file"}</small></span><button className="icon-button" aria-label="Remove selected file" onClick={() => { setSelectedFile(null); setAudioUrl(""); setResult(null); }}><X size={16}/></button></div>}
              {audioUrl && <audio className="audio-player" controls src={audioUrl} aria-label="Preview selected recording"/>}
              <div className="consent-box"><div className="consent-title"><LockKeyhole size={16}/><b>Consent & data handling</b></div><label className="consent-option"><input type="checkbox" checked={consent} onChange={event => setConsent(event.target.checked)}/><span>I have permission to submit this recording and consent to temporary analysis by the configured VGRS API.</span></label><p>Recordings are not intentionally saved in persistent VGRS history or used for model training. Multipart processing may use temporary OS storage.</p></div>
              <button className="button button-primary action-button" disabled={!selectedFile || !consent || busy} onClick={analyzeOne}>{busy ? <LoaderCircle className="spin" size={17}/> : <Sparkles size={17}/>} {busy ? "Analyzing recording…" : "Run acoustic analysis"} {!busy && <ArrowRight size={17}/>}</button><div className="panel-hint"><ShieldCheck size={14}/> Processed by the configured API endpoint.</div>
            </div>
            <div className="panel report-panel"><PanelHeading step="02" title="Analysis report" subtitle="Measured acoustic features and model availability." meta={result ? "COMPLETE" : "AWAITING INPUT"}/>{result ? <><ReportSummary result={result} onDetails={() => document.getElementById("feature-details")?.scrollIntoView({ behavior: "smooth" })}/>{selectedFile && <ResearchContributionPanel enabled={Boolean(models?.research_contributions_enabled)} file={selectedFile} modelVersion={result.age_group.model_version} predictedAgeGroup={result.age_group.label}/>}</> : <EmptyReport busy={busy} onNotes={() => setShowResearchNotes(true)}/>}</div>
          </section>
          {result && <section className="panel features-panel" id="feature-details"><SectionHeading title="Acoustic feature breakdown" subtitle="Measurements returned by the audio pipeline; unavailable fields are not estimated." action={<span className={"quality-tag " + (result.audio_quality.status === "acceptable" ? "good" : "warn")}>{result.audio_quality.status.replace(/_/g, " ")}</span>}/>
            {result.audio_quality.warnings.length > 0 && <div className="quality-notes"><b><AlertCircle size={15}/> Recording quality notes</b>{result.audio_quality.warnings.map(warning => <p key={warning}>{warning}</p>)}</div>}
            <div className="feature-grid">{featureCards.map(card => <FeatureCard key={card.key} icon={card.icon} title={card.name} value={card.value} unit={card.unit} description={card.note}/>)}</div>
            <div className="metadata-grid"><MetaValue label="Duration" value={fmt(result.acoustic_features.duration_seconds, " s")}/><MetaValue label="Output sample rate" value={fmt(result.acoustic_features.sample_rate_hz, " Hz")}/><MetaValue label="Source sample rate" value={fmt(result.acoustic_features.source_sample_rate_hz, " Hz")}/><MetaValue label="Channels" value={fmt(result.acoustic_features.channel_count)}/><MetaValue label="Pipeline version" value={result.model_info.feature_pipeline_version}/></div>
          </section>}
        </div>}

        {page === "batch" && <div className="batch-page"><section className="panel batch-input"><PanelHeading step="01" title="Batch input" subtitle="Select up to 20 recordings for one batch request." meta="MAX 20 FILES"/><input ref={batchRef} className="visually-hidden" type="file" multiple accept={ACCEPT} onChange={event => { setBatchFiles(Array.from(event.target.files || []).slice(0, 20)); setBatchRows([]); }} aria-label="Choose batch audio files"/><div className="batch-select-zone"><div className="upload-mark"><FileAudio size={23}/></div><h3>Build your analysis batch</h3><p>Each recording is validated and reported separately.</p><button className="button button-outline" onClick={() => batchRef.current?.click()}><Plus size={16}/> Select recordings</button></div>
          {batchFiles.length > 0 && <div className="batch-file-list"><div className="batch-list-heading"><b>Selected recordings <span>{batchFiles.length}</span></b><button className="text-button" onClick={() => { setBatchFiles([]); setBatchRows([]); }}>Clear all</button></div>{batchFiles.map((file, index) => <div className="batch-file-item" key={file.name + file.size}><span className="batch-file-symbol"><FileAudio size={16}/></span><span className="batch-file-copy"><b>{file.name}</b><small>{(file.size / 1048576).toFixed(2)} MB</small></span><small className="batch-index">{String(index + 1).padStart(2, "0")}</small><button className="icon-button" aria-label={"Remove " + file.name} onClick={() => setBatchFiles(old => old.filter((_, i) => i !== index))}><X size={15}/></button></div>)}</div>}
          <div className="consent-box"><div className="consent-title"><LockKeyhole size={16}/><b>Explicit consent required</b></div><label className="consent-option"><input type="checkbox" checked={consent} onChange={event => setConsent(event.target.checked)}/><span>I have permission for every selected recording and consent to temporary analysis.</span></label></div>
          <button className="button button-primary action-button" disabled={!batchFiles.length || !consent || batchBusy} onClick={analyzeBatch}>{batchBusy ? <LoaderCircle className="spin" size={17}/> : <Sparkles size={17}/>} {batchBusy ? "Processing batch…" : "Analyze " + batchFiles.length + " recordings"} {!batchBusy && <ArrowRight size={17}/>}</button>
        </section>
        <section className="panel batch-result-panel"><PanelHeading step="02" title="Batch results" subtitle="Per-file status, signal measures and export." meta={batchRows.length ? batchRows.length + " ROWS" : "NO RESULTS"}/>{batchRows.length === 0 ? <div className="batch-empty"><span className="batch-empty-icon"><BarChart3 size={25}/></span><h3>Your batch report will appear here</h3><p>Run a batch to see validation results, acoustic measurements and individual errors. Partial failures remain visible instead of cancelling the whole report.</p><div className="empty-points"><span><Check size={13}/> Per-file validation</span><span><Check size={13}/> Partial failures</span><span><Check size={13}/> CSV / JSON export</span></div></div> : <><div className="batch-summary"><div><small>Processed</small><b>{batchRows.length}</b></div><div><small>Completed</small><b className="green-text">{batchRows.filter(row => row.status === "completed").length}</b></div><div><small>Failed</small><b className="red-text">{batchRows.filter(row => row.status !== "completed").length}</b></div></div><div className="batch-results-heading"><b>Individual recordings</b><span><button className="button button-quiet button-small" onClick={() => exportBatch("csv")}><FileText size={14}/> Export CSV</button><button className="button button-quiet button-small" onClick={() => exportBatch("json")}>Export JSON</button></span></div><div className="responsive-table"><table><thead><tr><th>Recording</th><th>Status</th><th>Duration</th><th>Median F0</th><th>Quality</th><th>Classifier</th></tr></thead><tbody>{batchRows.map((row, index) => <tr key={row.filename + index}><td className="filename-cell">{row.filename}</td><td><StatusPill tone={row.status === "completed" ? "good" : "bad"}>{row.status}</StatusPill></td><td>{fmt(row.result?.acoustic_features.duration_seconds, " s")}</td><td>{fmt(row.result?.acoustic_features.f0_median_hz, " Hz")}</td><td>{row.result?.audio_quality.status.replace(/_/g, " ") || "—"}</td><td>{row.result?.vocal_presentation.status === "model_unavailable" ? "Unavailable" : "—"}</td></tr>)}</tbody></table></div></>}</section></div>}

        {page === "history" && <section className="panel history-panel"><SectionHeading title="Recent session activity" subtitle="Only analyses completed during this browser session are listed." action={<button className="button button-quiet button-small" disabled={!activities.length} onClick={clearSession}><RefreshCw size={14}/> Clear session</button>}/>{activities.length === 0 ? <EmptyState icon={History} title="No analysis history yet" text="Completed reports will appear here for this session. A persistent history database is not configured." action="Start an analysis" onAction={() => navigate("analyze")}/> : <ActivityTable items={activities} onOpen={item => { setResult(item.result); setSelectedFile(null); setAudioUrl(""); setPage("analyze"); setNotice("Viewing a report held in the current browser session."); }}/>}</section>}

        {page === "settings" && <div className="settings-page"><section className="panel settings-panel"><SectionHeading title="Service connection" subtitle="Live connectivity from your running API." action={<button className="button button-quiet button-small" onClick={() => { setHealth({ status: "checking" }); Promise.all([fetch(API + "/api/v1/health"), fetch(API + "/api/v1/models")]).then(async ([a,b]) => { if (!a.ok || !b.ok) throw new Error("Offline"); const h = await a.json(); const m = await b.json(); setHealth({ status: "online", version: h.version, classificationReady: Boolean(h.classification_ready) }); setModels(m); }).catch(() => setHealth({ status: "offline" })); }}><RefreshCw size={14}/> Recheck</button>}/><SettingRow title="API endpoint" description="Destination used for health checks and audio requests."><code>{API}</code></SettingRow><SettingRow title="Health status" description="Latest response from the service health endpoint."><StatusPill tone={health.status === "online" ? "good" : health.status === "offline" ? "bad" : "neutral"}>{health.status}</StatusPill></SettingRow><SettingRow title="Application version" description="Version reported by the API."><b>{health.version || "—"}</b></SettingRow></section>
          <section className="panel settings-panel"><SectionHeading title="Model availability" subtitle="Unavailable models do not produce guesses."/><ModelRow icon={AudioLines} title="VGRS acoustic baseline" subtitle={"Feature pipeline · " + (acousticModel?.version || "version unknown")} detail="Pitch summary, spectral features, energy and recording-quality indicators." status={acousticModel?.status || "unknown"} ready={acousticModel?.status === "available"}/><ModelRow icon={Fingerprint} title="Age & vocal-presentation classifier" subtitle={classifierModel?.id || "Classification adapter"} detail={classifierModel?.limitations?.[0] || "A suitable licensed, validated classifier is not loaded."} status={classifierModel?.status || "unknown"} ready={classifierModel?.status === "available"}/></section>
          <section className="panel settings-panel"><SectionHeading title="Privacy and safeguards" subtitle="Implementation behavior, not a certification."/><div className="privacy-items"><PrivacyItem good title="Explicit consent" text="Both UI and API require consent before analysis."/><PrivacyItem good title="No intentional persistent audio retention" text="No persistent audio-history database is configured. Multipart parsing may use operating-system temporary storage."/><PrivacyItem good title="No model-training reuse" text="Submitted recordings are not reused for model training."/><PrivacyItem title="Local research build" text="Public deployment hardening and classifier accuracy validation remain outstanding."/></div></section></div>}
      </main>
      <footer className="app-footer"><span><b>VGRS</b><i/> Voice Analysis Studio</span><span><span className={"footer-led " + health.status}/>{health.status === "online" ? "Local research service" : "Local environment"}<span className="footer-version">v{health.version || "0.1.0"}</span></span></footer>
    </div>
  </div>;
}

function NavButton({ icon: Icon, label, active, collapsed, count, onClick }: { icon: LucideIcon; label: string; active: boolean; collapsed: boolean; count?: number; onClick: () => void }) {
  return <button className={"nav-item" + (active ? " active" : "") + (collapsed ? " only-icon" : "")} title={collapsed ? label : undefined} aria-current={active ? "page" : undefined} onClick={onClick}><Icon size={18}/>{!collapsed && <><span>{label}</span>{count ? <small>{count}</small> : null}</>}</button>;
}
function PanelHeading({ step, title, subtitle, meta }: { step: string; title: string; subtitle: string; meta?: string }) {
  return <div className="panel-heading"><span className="step-number">{step}</span><div className="panel-heading-copy"><h2>{title}</h2><p>{subtitle}</p></div>{meta && <span className="panel-meta">{meta}</span>}</div>;
}
function SectionHeading({ title, subtitle, badge, action }: { title: string; subtitle: string; badge?: string; action?: ReactNode }) {
  return <div className="section-heading"><div><h2>{title}</h2><p>{subtitle}</p></div>{badge && <span className="subtle-badge">{badge}</span>}{action && <div className="section-action">{action}</div>}</div>;
}
function StatusPill({ tone, children }: { tone: "good" | "warn" | "bad" | "neutral"; children: ReactNode }) {
  return <span className={"status-pill " + tone}><i/>{children}</span>;
}
function Callout({ kind, icon: Icon, title, children, onClose }: { kind: "warning" | "error" | "success"; icon: LucideIcon; title: string; children: ReactNode; onClose: () => void }) {
  return <div className={"callout " + kind}><Icon size={18}/><div><b>{title}</b><p>{children}</p></div><button className="icon-button" aria-label="Dismiss message" onClick={onClose}><X size={15}/></button></div>;
}
function StatCard({ label, value, description, icon: Icon, tint, footer }: { label: string; value: string; description: string; icon: LucideIcon; tint: string; footer: string }) {
  return <article className="stat-card"><div className="stat-top"><span>{label}</span><span className={"stat-icon " + tint}><Icon size={17}/></span></div><strong>{value}</strong><p>{description}</p><div className="stat-foot"><span className={"state-point " + tint}/>{footer}</div></article>;
}
function WorkflowRow({ icon: Icon, tone, title, description, tags, onClick }: { icon: LucideIcon; tone: string; title: string; description: string; tags: string[]; onClick: () => void }) {
  return <button className="workflow-row" onClick={onClick}><span className={"workflow-icon " + tone}><Icon size={20}/></span><span className="workflow-copy"><b>{title}</b><small>{description}</small><span className="workflow-tags">{tags.map(tag => <i key={tag}>{tag}</i>)}</span></span><ArrowRight className="workflow-arrow" size={17}/></button>;
}
function Readiness({ icon: Icon, title, detail, value, tone }: { icon: LucideIcon; title: string; detail: string; value: string; tone: "good" | "warn" | "bad" | "neutral" }) {
  return <div className="readiness-row"><span className="readiness-icon"><Icon size={16}/></span><span className="readiness-copy"><b>{title}</b><small>{detail}</small></span><StatusPill tone={tone}>{value}</StatusPill></div>;
}
function EmptyReport({ busy, onNotes }: { busy: boolean; onNotes: () => void }) {
  return <div className="empty-report"><div className="report-art"><span className="report-orbit one"/><span className="report-orbit two"/><span className="report-orbit three"/><AudioLines size={28}/></div><h3>{busy ? "Analyzing recording" : "Your report starts here"}</h3><p>{busy ? "Validating the audio and extracting real acoustic measurements." : "Select a recording to create a report from measurements returned by the VGRS API."}</p>{busy ? <div className="progress-line"><span/></div> : <div className="report-checks"><span><Check size={13}/> Input validation</span><span><Check size={13}/> Signal features</span><span><Check size={13}/> Model status</span></div>}<div className="not-enabled-card"><span><Fingerprint size={17}/></span><div><b>Classification needs model approval</b><small>A licensed, validated model must be configured before estimates can be treated as production output.</small></div><StatusPill tone="warn">Gated</StatusPill></div>{!busy && <button className="learn-link" onClick={onNotes}>Review scientific limitations <ArrowRight size={14}/></button>}</div>;
}
function ReportSummary({ result, onDetails }: { result: AnalysisResult; onDetails: () => void }) {
  return <div className="report-summary"><div className="report-success"><CheckCircle2 size={18}/><span><b>Acoustic analysis complete</b><small>Report {reportId(result.analysis_id)} · {result.filename}</small></span><StatusPill tone={result.audio_quality.status === "acceptable" ? "good" : "warn"}>{result.audio_quality.status.replace(/_/g, " ")}</StatusPill></div>
    <PredictionRow icon={Fingerprint} title="Vocal presentation" prediction={result.vocal_presentation}/><PredictionRow icon={Clock3} title="Age-group estimate" prediction={result.age_group}/>
    <div className="quick-values"><div><small>Duration</small><b>{fmt(result.acoustic_features.duration_seconds, " s")}</b></div><div><small>Median F0</small><b>{fmt(result.acoustic_features.f0_median_hz, " Hz")}</b></div><div><small>Quality</small><b>{result.audio_quality.status === "acceptable" ? "Acceptable" : "Review"}</b></div></div>
    <div className="report-note"><Info size={15}/><p>{result.model_info.model_enabled ? (result.model_info.production_approved ? "Classifier output is enabled under the model and population scope documented by its validation report; estimates are not verified identity or chronological age." : "Experimental classifier output is enabled but has not been validated for production use or all target populations.") : "This build reports measured acoustic features only; age and vocal-presentation estimates are withheld."}</p></div>
    {result.audio_quality.warnings.length > 0 && <div className="quality-notes"><b><AlertCircle size={15}/> Recording quality notes</b>{result.audio_quality.warnings.map(warning => <p key={warning}>{warning}</p>)}</div>}
    <button className="button button-primary full-button" onClick={onDetails}>Review acoustic measurements <ArrowRight size={16}/></button>
  </div>;
}
function PredictionRow({ icon: Icon, title, prediction }: { icon: LucideIcon; title: string; prediction: Prediction }) {
  return <div className="prediction-row"><span className="prediction-icon"><Icon size={17}/></span><span className="prediction-copy"><small>{title.toUpperCase()}</small><b>{prediction.label || "Not enabled"}</b><p>{prediction.note}</p></span><StatusPill tone="warn">{prediction.status === "model_unavailable" ? "Unavailable" : prediction.status}</StatusPill></div>;
}
function FeatureCard({ icon: Icon, title, value, unit, description }: { icon: LucideIcon; title: string; value: number | null | undefined; unit: string; description: string }) {
  return <article className="feature-card"><div className="feature-top"><span>{title}</span><span className="feature-icon"><Icon size={16}/></span></div><strong>{fmt(value, unit)}</strong><p>{description}</p></article>;
}
function MetaValue({ label, value }: { label: string; value: string }) {
  return <div className="meta-value"><small>{label}</small><b>{value}</b></div>;
}
function ActivityTable({ items, onOpen }: { items: ActivityItem[]; onOpen: (item: ActivityItem) => void }) {
  return <div className="activity-table"><div className="activity-head"><span>RECORDING</span><span>ANALYZED</span><span>QUALITY</span><span>CLASSIFICATION</span><span/></div>{items.map(item => <button className="activity-row" key={item.id} onClick={() => onOpen(item)}><span className="activity-file"><span><FileAudio size={16}/></span><span><b>{item.filename}</b><small>Report {reportId(item.id)}</small></span></span><span className="activity-date">{timestamp(item.createdAt)}</span><span><StatusPill tone={item.result.audio_quality.status === "acceptable" ? "good" : "warn"}>{item.result.audio_quality.status.replace(/_/g, " ")}</StatusPill></span><span className="muted-cell">Not enabled</span><ChevronRight size={16}/></button>)}</div>;
}
function EmptyState({ icon: Icon, title, text, action, onAction }: { icon: LucideIcon; title: string; text: string; action: string; onAction: () => void }) {
  return <div className="empty-state"><span><Icon size={23}/></span><h3>{title}</h3><p>{text}</p><button className="button button-primary" onClick={onAction}>{action}<ArrowRight size={15}/></button></div>;
}
function SettingRow({ title, description, children }: { title: string; description: string; children: ReactNode }) {
  return <div className="setting-row"><div><b>{title}</b><small>{description}</small></div><span>{children}</span></div>;
}
function ModelRow({ icon: Icon, title, subtitle, detail, status, ready }: { icon: LucideIcon; title: string; subtitle: string; detail: string; status: string; ready: boolean }) {
  return <div className="model-row"><span className={"model-icon " + (ready ? "ready" : "disabled")}><Icon size={19}/></span><span className="model-copy"><b>{title}</b><small>{subtitle}</small><p>{detail}</p></span><StatusPill tone={ready ? "good" : "warn"}>{status}</StatusPill></div>;
}
function PrivacyItem({ good, title, text }: { good?: boolean; title: string; text: string }) {
  return <div className="privacy-item"><span className={good ? "privacy-good" : "privacy-warn"}>{good ? <CheckCircle2 size={16}/> : <AlertCircle size={16}/>}</span><span><b>{title}</b><small>{text}</small></span></div>;
}


function ResearchContributionPanel({ enabled, file, modelVersion, predictedAgeGroup }: {
  enabled: boolean;
  file: File;
  modelVersion: string | null;
  predictedAgeGroup: string | null;
}) {
  const [consentToResearch, setConsentToResearch] = useState(false);
  const [actualAge, setActualAge] = useState("");
  const [participantCode, setParticipantCode] = useState("");
  const [busy, setBusy] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [contributionId, setContributionId] = useState("");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  async function contribute() {
    const age = Number(actualAge);
    if (!consentToResearch) { setError("Separate research consent is required."); return; }
    if (!Number.isInteger(age) || age < 18 || age > 110) { setError("Enter the adult speaker's age in whole years (18–110)."); return; }
    if (participantCode.trim() && !/^[A-Za-z0-9_-]{4,40}$/.test(participantCode.trim())) {
      setError("Participant code must be 4–40 letters, digits, underscores or hyphens. Do not use a name or email.");
      return;
    }
    setBusy(true); setError(""); setMessage("");
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("consent_to_research", "true");
      form.append("actual_age_years", String(age));
      form.append("participant_code", participantCode.trim());
      const response = await fetch(API + "/api/v1/research/contributions", { method: "POST", body: form });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(body?.detail?.message || "The research contribution could not be saved.");
      setContributionId(body.contribution_id);
      setMessage("Contribution stored for offline evaluation. The current model has not been updated.");
      setConsentToResearch(false);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Unable to submit research contribution.");
    } finally { setBusy(false); }
  }

  async function removeContribution() {
    if (!contributionId) return;
    setDeleting(true); setError(""); setMessage("");
    try {
      const response = await fetch(API + "/api/v1/research/contributions/" + encodeURIComponent(contributionId), { method: "DELETE" });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body?.detail?.message || "The contribution could not be deleted.");
      }
      setContributionId("");
      setMessage("The stored research contribution and recording were deleted.");
      setActualAge("");
      setParticipantCode("");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Unable to delete research contribution.");
    } finally { setDeleting(false); }
  }

  return <section className="research-contribution">
    <div className="research-contribution-heading"><span className="research-contribution-icon"><ShieldCheck size={17}/></span><div><b>Help evaluate future age estimates</b><small>Optional · separate consent · adult participants only</small></div><StatusPill tone="neutral">Research</StatusPill></div>
    <p className="research-contribution-copy">With explicit consent, VGRS can keep a copy of this recording and a self-reported adult age label on the configured API server for offline model evaluation. It does not change predictions automatically. Only contribute to a server you control and never submit a child's recording.</p>
    {message && <div className="research-message"><CheckCircle2 size={15}/><span>{message}{contributionId && <small>Contribution ID: <code>{contributionId}</code></small>}</span></div>}
    {error && <div className="research-error"><AlertCircle size={15}/><span>{error}</span></div>}
    {!enabled && <div className="research-disabled-note"><AlertCircle size={15}/><span>Research collection is disabled. It is off by default; enable it explicitly only in a local development environment you control.</span></div>}
    {enabled && !contributionId && <div className="research-contribution-form">
      <label className="research-field"><span>Adult speaker's actual age</span><input type="number" min="18" max="110" step="1" value={actualAge} onChange={event => setActualAge(event.target.value)} placeholder="18–110" disabled={busy}/></label>
      <label className="research-field"><span>Anonymous participant code (optional)</span><input type="text" maxLength={40} value={participantCode} onChange={event => setParticipantCode(event.target.value)} placeholder="e.g. SPEAKER_04" disabled={busy}/><small>Use the same non-identifying code for this speaker's recordings. Never enter a name or email.</small></label>
      <label className="research-consent"><input type="checkbox" checked={consentToResearch} onChange={event => setConsentToResearch(event.target.checked)} disabled={busy}/><span>I have permission to retain this adult speaker's recording and self-reported age for research evaluation. I understand this stores a copy on the configured API server and does not instantly train or change the model.</span></label>
      <button className="button button-outline research-submit" disabled={!consentToResearch || !actualAge || busy} onClick={contribute}>{busy ? <LoaderCircle size={15} className="spin"/> : <UploadCloud size={15}/>} {busy ? "Saving contribution…" : "Contribute for research"}</button>
      {modelVersion && <small className="research-model-version">Associated classifier: {modelVersion} · Current predicted band: {predictedAgeGroup || "unavailable"}</small>}
    </div>}
    {enabled && contributionId && <button className="button button-quiet research-delete" disabled={deleting} onClick={removeContribution}>{deleting ? <LoaderCircle size={15} className="spin"/> : <X size={15}/>} {deleting ? "Deleting…" : "Delete my contribution"}</button>}
  </section>;
}
