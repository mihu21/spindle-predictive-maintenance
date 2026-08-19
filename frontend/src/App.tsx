import { useQuery } from "@tanstack/react-query";
import { NavLink, Route, Routes, useParams } from "react-router-dom";
import { formatHours, formatTime, getJson } from "./api";
import type { Lifecycle, Machine, OperatingContext, Overview, PlantMetric, VibrationOperatingInference } from "./types";

const nav = [
  ["Overview", "/"], ["Machines", "/machines"], ["Lifecycles", "/lifecycles"],
  ["Operating context", "/operating-context"], ["Vibration inference", "/vibration-operating"], ["Sources", "/sources"], ["Plant evidence", "/performance"], ["Model", "/model"],
  ["System", "/system"], ["Audit", "/audit"],
];

function QueryState({ loading, error }: { loading: boolean; error: Error | null }) {
  if (loading) return <div className="notice">Loading committed shadow evidence…</div>;
  if (error) return <div className="notice error">API unavailable: {error.message}</div>;
  return null;
}

function Badge({ value = "UNKNOWN" }: { value?: string | null }) {
  const resolved = value ?? "UNKNOWN";
  return <span className={`badge ${resolved.toLowerCase().replaceAll("_", "-")}`}>{resolved}</span>;
}

function MetricCard({ label, value, note }: { label: string; value: string | number; note?: string }) {
  return <article className="metric"><span>{label}</span><strong>{value}</strong>{note && <small>{note}</small>}</article>;
}

function EnvironmentBanner() {
  const query = useQuery({ queryKey: ["overview"], queryFn: () => getJson<Overview>("/overview") });
  if (query.data?.environment_mode !== "DEMO") return null;
  return <div className="demo-banner" role="status"><strong>LOCAL DEMO MODE</strong><span>Synthetic plant-shadow evidence</span><span>Production authorization: DISABLED</span></div>;
}

function MachineTable({ machines }: { machines: Machine[] }) {
  return <div className="table-wrap"><table><thead><tr><th>Machine</th><th>Source / line</th><th>Equipment</th><th>Manufacturer</th><th>Model</th><th>Forecast</th><th>WARNING</th><th>CRITICAL</th><th>Exposure</th><th>Last seen</th></tr></thead>
    <tbody>{machines.map(machine => <tr key={machine.machine_uid}>
      <td><NavLink to={`/machines/${encodeURIComponent(machine.machine_uid)}`}>{machine.machine_id}</NavLink><small>{machine.demo_scenarios ?? machine.machine_uid}</small></td>
      <td>{machine.source_key}<small>{machine.line_sel}</small></td>
      <td><Badge value={machine.operating_state} /><small>{machine.operating_state_source ?? "UNAVAILABLE"}</small></td>
      <td><Badge value={machine.health_state_manufacturer ?? "UNKNOWN"} /></td>
      <td><Badge value={machine.health_state_model} /></td>
      <td><Badge value={machine.forecast_state} /><small>{machine.forecast_reason}</small></td>
      <td>{formatHours(machine.warning_point_hours)}</td><td>{formatHours(machine.critical_point_hours)}</td>
      <td>{machine.cumulative_operating_hours == null ? "Unavailable" : `${machine.cumulative_operating_hours.toFixed(1)} h`}</td>
      <td>{formatTime(machine.latest_seen_at)}</td>
    </tr>)}</tbody></table></div>;
}

function OverviewPage() {
  const overview = useQuery({ queryKey: ["overview"], queryFn: () => getJson<Overview>("/overview") });
  const machines = useQuery({ queryKey: ["machines"], queryFn: () => getJson<{ items: Machine[] }>("/machines?limit=20") });
  const error = (overview.error || machines.error) as Error | null;
  if (!overview.data || !machines.data) return <QueryState loading={overview.isLoading || machines.isLoading} error={error} />;
  return <><header className="page-head"><div><p className="eyebrow">Plant shadow / observational only</p><h2>Operations overview</h2></div><Badge value="PRODUCTION_NOT_AUTHORIZED" /></header>
    <section className="metrics">
      <MetricCard label="Machines" value={overview.data.machines} note={`${overview.data.sources} read-only source(s)`} />
      <MetricCard label="Active lifecycles" value={overview.data.active_lifecycles} />
      <MetricCard label="Confirmed running" value={overview.data.machines_running} />
      <MetricCard label="Paused / uncertain" value={overview.data.machines_paused} note="OFF, IDLE, MAINTENANCE, or UNKNOWN" />
      <MetricCard label="Vibration calibrating" value={overview.data.machines_vibration_calibrating} note="RUL remains paused" />
      <MetricCard label="Vibration-confirmed" value={overview.data.machines_vibration_inferred_running} note="High-confidence production pattern" />
      <MetricCard label="Forecasts available" value={overview.data.forecasts_available} />
      <MetricCard label="Forecasts withheld" value={overview.data.forecasts_withheld} note="Unavailable values are cleared" />
      <MetricCard label="Committed predictions" value={overview.data.predictions} />
      <MetricCard label="Active alerts" value={overview.data.active_alerts} />
    </section>
    <section className="panel"><div className="panel-title"><h3>Machine priority</h3><span>Manufacturer safety first</span></div><MachineTable machines={machines.data.items} /></section>
  </>;
}

function MachinesPage() {
  const query = useQuery({ queryKey: ["machines-all"], queryFn: () => getJson<{ items: Machine[] }>("/machines?limit=1000") });
  if (!query.data) return <QueryState loading={query.isLoading} error={query.error as Error | null} />;
  return <><header className="page-head"><div><p className="eyebrow">Independent streams</p><h2>Machines</h2></div><span>{query.data.items.length} machine(s)</span></header><MachineTable machines={query.data.items} /></>;
}

function MachinePage() {
  const { machineUid = "" } = useParams();
  const query = useQuery({ queryKey: ["machine", machineUid], queryFn: () => getJson<any>(`/machines/${encodeURIComponent(machineUid)}`) });
  const sensors = useQuery({ queryKey: ["sensors", machineUid], queryFn: () => getJson<{ items: any[] }>(`/machines/${encodeURIComponent(machineUid)}/sensors?limit=120`) });
  if (!query.data) return <QueryState loading={query.isLoading} error={query.error as Error | null} />;
  const latest = query.data.latest_prediction;
  const context = query.data.latest_operating_context as OperatingContext | null;
  const vibration = query.data.latest_vibration_operating_inference as VibrationOperatingInference | null;
  const scenarioNote = query.data.demo_scenarios?.map((item: any) => item.scenario_name).join(", ");
  return <><header className="page-head"><div><p className="eyebrow">{query.data.machine.source_key} / {query.data.machine.line_sel}</p><h2>{query.data.machine.machine_id}</h2><small>{scenarioNote || machineUid}</small></div><div className="badge-stack"><Badge value={context?.operating_state} /><Badge value={latest?.health_state_manufacturer ?? latest?.health_state_model} /></div></header>
    <section className="metrics"><MetricCard label="WARNING operating RUL" value={formatHours(latest?.warning_point_hours)} note={latest?.warning_withhold_reason} /><MetricCard label="CRITICAL operating RUL" value={formatHours(latest?.critical_point_hours)} note={latest?.critical_withhold_reason} /><MetricCard label="Forecast" value={latest?.forecast_state ?? "UNAVAILABLE"} note={latest?.warning_withhold_reason ?? latest?.critical_withhold_reason ?? "No committed forecast"} /><MetricCard label="Runtime admitted" value={context?.admitted_to_runtime ? "YES" : "NO"} note={context?.reason_code} /><MetricCard label="Equipment state" value={context?.operating_state ?? "UNKNOWN"} note={`${context?.operating_state_source ?? "UNAVAILABLE"} / ${context ? `${(context.operating_state_confidence * 100).toFixed(0)}%` : "0%"}`} /><MetricCard label="Vibration inference" value={vibration?.classification ?? "CALIBRATING"} note={`${vibration?.calibration_state ?? "CALIBRATING"} / ${vibration ? `${(vibration.confidence * 100).toFixed(1)}%` : "0%"}`} /><MetricCard label="Operating exposure" value={context?.cumulative_operating_seconds == null ? "Unavailable" : `${(context.cumulative_operating_seconds / 3600).toFixed(1)} h`} note={context?.reason_code} /><MetricCard label="Lifecycle" value={query.data.current_lifecycle?.status ?? "UNINITIALIZED"} note={query.data.current_lifecycle?.left_censored ? "Left-censored" : undefined} /></section>
    <section className="panel"><div className="panel-title"><h3>Recent sensor evidence</h3><span>All rows retained; only confirmed RUNNING enters prognostics</span></div><div className="table-wrap"><table><thead><tr><th>Event time</th><th>Equipment</th><th>Vibration inference</th><th>Model admitted</th><th>Operating hours</th><th>VRMS</th><th>ARMS</th><th>APEAK</th><th>Crest</th><th>Temp</th></tr></thead><tbody>{sensors.data?.items.map(row => <tr key={row.ingestion_id}><td>{formatTime(row.event_timestamp)}</td><td><Badge value={row.operating_state} /><small>{row.operating_context_reason}</small></td><td><Badge value={row.vibration_classification ?? "UNAVAILABLE"} /><small>{row.vibration_reason}</small></td><td>{row.admitted_to_runtime ? "YES" : "NO"}</td><td>{row.cumulative_operating_hours?.toFixed(2) ?? "Unavailable"}</td><td>{row.vrms.toFixed(2)}</td><td>{row.arms.toFixed(2)}</td><td>{row.apeak.toFixed(2)}</td><td>{row.crest.toFixed(2)}</td><td>{row.temp.toFixed(1)} °C</td></tr>)}</tbody></table></div></section>
  </>;
}

function LifecyclesPage() {
  const query = useQuery({ queryKey: ["lifecycles"], queryFn: () => getJson<{ items: Lifecycle[] }>("/lifecycles?limit=500") });
  if (!query.data) return <QueryState loading={query.isLoading} error={query.error as Error | null} />;
  return <><header className="page-head"><div><p className="eyebrow">Auditable segmentation</p><h2>Lifecycles</h2></div></header><div className="table-wrap"><table><thead><tr><th>Lifecycle</th><th>Machine</th><th>Status</th><th>Observed start</th><th>End</th><th>Censoring</th><th>Closure</th></tr></thead><tbody>{query.data.items.map(l => <tr key={l.lifecycle_id}><td>{l.lifecycle_id}</td><td>{l.machine_uid}</td><td><Badge value={l.status} /></td><td>{formatTime(l.observed_start_timestamp)}</td><td>{formatTime(l.end_timestamp)}</td><td>{l.left_censored ? "LEFT" : "NO"}</td><td>{l.closure_reason ?? "—"}<small>{l.closure_confidence}</small></td></tr>)}</tbody></table></div></>;
}

function SourcesPage() {
  const query = useQuery({ queryKey: ["sources"], queryFn: () => getJson<{ items: any[]; access: string }>("/sources") });
  if (!query.data) return <QueryState loading={query.isLoading} error={query.error as Error | null} />;
  return <><header className="page-head"><div><p className="eyebrow">Credentials remain backend-only</p><h2>Data sources</h2></div><Badge value={query.data.access} /></header><div className="card-grid">{query.data.items.map(source => <article className="panel" key={source.source_key}><h3>{source.source_key}</h3><p>Configuration v{source.version_number}</p><dl><dt>Enabled</dt><dd>{source.enabled ? "Yes" : "No"}</dd><dt>Archived</dt><dd>{source.archived ? "Yes" : "No"}</dd><dt>Credentials exposed</dt><dd>No</dd></dl></article>)}</div></>;
}

function PerformancePage() {
  const query = useQuery({ queryKey: ["performance"], queryFn: () => getJson<{ warning: PlantMetric; critical: PlantMetric }>("/plant-evaluation") });
  if (!query.data) return <QueryState loading={query.isLoading} error={query.error as Error | null} />;
  const Target = ({ value }: { value: PlantMetric }) => <article className="panel target"><div className="panel-title"><h3>{value.target}</h3><Badge value={value.status} /></div><div className="metrics compact"><MetricCard label="Lifecycles" value={value.counts.lifecycles} /><MetricCard label="Serviceable points" value={value.counts.serviceable_predictions} /><MetricCard label="Availability" value={value.availability == null ? "Unavailable" : `${(value.availability * 100).toFixed(1)}%`} /></div>{value.point_metrics ? <pre>{JSON.stringify(value.point_metrics, null, 2)}</pre> : <p className="empty">Waiting for sufficient independently eligible exact target evidence.</p>}</article>;
  return <><header className="page-head"><div><p className="eyebrow">Stored online predictions only</p><h2>Plant evidence</h2></div><Badge value="SHADOW_ONLY" /></header><div className="target-grid"><Target value={query.data.warning} /><Target value={query.data.critical} /></div></>;
}

function JsonPage({ title, path }: { title: string; path: string }) {
  const query = useQuery({ queryKey: [path], queryFn: () => getJson<any>(path) });
  if (!query.data) return <QueryState loading={query.isLoading} error={query.error as Error | null} />;
  return <><header className="page-head"><div><p className="eyebrow">Read-only evidence</p><h2>{title}</h2></div></header><pre className="json panel">{JSON.stringify(query.data, null, 2)}</pre></>;
}

export default function App() {
  return <div className="shell"><aside><div className="brand"><span>VVB001</span><strong>Plant Shadow</strong><small>Observe. Preserve. Verify.</small></div><nav>{nav.map(([label, path]) => <NavLink key={path} to={path} end={path === "/"}>{label}</NavLink>)}</nav><div className="side-foot"><i />Local read-only UI<br />No plant control</div></aside><main><EnvironmentBanner /><Routes><Route path="/" element={<OverviewPage />} /><Route path="/machines" element={<MachinesPage />} /><Route path="/machines/:machineUid" element={<MachinePage />} /><Route path="/lifecycles" element={<LifecyclesPage />} /><Route path="/operating-context" element={<JsonPage title="Operating-state evidence" path="/operating-state-evidence?limit=500" />} /><Route path="/vibration-operating" element={<JsonPage title="Vibration-derived operating inference" path="/vibration-operating?limit=500" />} /><Route path="/sources" element={<SourcesPage />} /><Route path="/performance" element={<PerformancePage />} /><Route path="/model" element={<JsonPage title="Model identity" path="/model" />} /><Route path="/system" element={<JsonPage title="System health" path="/system/health" />} /><Route path="/audit" element={<JsonPage title="Audit trail" path="/audit?limit=200" />} /></Routes></main></div>;
}
