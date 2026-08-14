export type Overview = {
  sources: number;
  machines: number;
  active_lifecycles: number;
  predictions: number;
  forecasts_available: number;
  forecasts_withheld: number;
  machines_running: number;
  machines_paused: number;
  machines_vibration_calibrating: number;
  machines_vibration_inferred_running: number;
  active_alerts: number;
  validation_domain: string;
  plant_production_authorized: boolean;
};

export type Machine = {
  machine_uid: string;
  source_key: string;
  line_sel: string;
  machine_id: string;
  latest_seen_at: string;
  health_state_model?: string | null;
  health_state_manufacturer?: string | null;
  forecast_state?: string | null;
  warning_point_hours?: number | null;
  critical_point_hours?: number | null;
  operating_state?: string | null;
  operating_state_source?: string | null;
  operating_state_confidence?: number | null;
  admitted_to_runtime?: number | null;
  cumulative_operating_hours?: number | null;
  vibration_classification?: string | null;
  vibration_calibration_state?: string | null;
  vibration_confidence?: number | null;
};

export type VibrationOperatingInference = {
  classification: string;
  confidence: number;
  reason_code: string;
  calibration_state: string;
  sample_count: number;
  history_span_seconds: number;
  window_energy?: number | null;
  window_variability?: number | null;
  production_similarity?: number | null;
  novelty_score?: number | null;
  persistent_running_seconds: number;
  calibration_artifact_sha256?: string | null;
};

export type OperatingContext = {
  operating_state: string;
  operating_state_source: string;
  operating_state_confidence: number;
  admitted_to_runtime: number;
  reason_code: string;
  cumulative_operating_seconds?: number | null;
  maintenance_event_id?: string | null;
};

export type Lifecycle = {
  lifecycle_id: string;
  machine_uid: string;
  sequence_number: number;
  status: string;
  observed_start_timestamp: string;
  end_timestamp?: string | null;
  left_censored: number;
  closure_reason?: string | null;
  closure_confidence?: string | null;
};

export type PlantMetric = {
  target: string;
  status: string;
  counts: Record<string, number>;
  availability: number | null;
  point_metrics: Record<string, number> | null;
  interval_metrics: Record<string, number> | null;
};
