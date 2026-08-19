// @vitest-environment jsdom

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "./App";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("plant shadow presentation contract", () => {
  it("does not encode unavailable RUL as a number", async () => {
    const { formatHours } = await import("./api");
    expect(formatHours(null)).toBe("Unavailable");
    expect(formatHours(0)).toBe("0.0 operating h");
  });

  it("exposes the vibration operating evidence route", () => {
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      const body = url.includes("/overview") ? {
        sources: 0,
        machines: 0,
        active_lifecycles: 0,
        predictions: 0,
        forecasts_available: 0,
        forecasts_withheld: 0,
        machines_running: 0,
        machines_paused: 0,
        machines_vibration_calibrating: 0,
        machines_vibration_inferred_running: 0,
        active_alerts: 0,
        validation_domain: "plant_shadow",
        environment_mode: "PLANT_SHADOW",
        evidence_description: "Plant-shadow evidence",
        demo_run: null,
        plant_production_authorized: false,
      } : { items: [] };
      return { ok: true, json: async () => body } as Response;
    }));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <MemoryRouter initialEntries={["/"]}><App /></MemoryRouter>
      </QueryClientProvider>,
    );
    expect(screen.getByRole("link", { name: "Vibration inference" })).toHaveAttribute(
      "href", "/vibration-operating",
    );
  });

  it("shows an explicit demo banner and renders distinct running and paused machines", async () => {
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      const body = url.includes("/overview") ? {
        sources: 1, machines: 3, active_lifecycles: 3, predictions: 72,
        forecasts_available: 8, forecasts_withheld: 64, machines_running: 2,
        machines_paused: 1, machines_vibration_calibrating: 0,
        machines_vibration_inferred_running: 0, active_alerts: 0,
        validation_domain: "plant_shadow", environment_mode: "DEMO",
        evidence_description: "Synthetic plant-shadow evidence", demo_run: { seed: 42 },
        plant_production_authorized: false,
      } : { items: [
        { machine_uid: "demo::healthy", machine_id: "DEMO_HEALTHY", source_key: "local-demo", line_sel: "DEMO_LINE", latest_seen_at: "2026-01-01T00:00:00Z", operating_state: "RUNNING", operating_state_source: "SYNTHETIC_FIXTURE", admitted_to_runtime: 1, forecast_state: "WITHHELD", forecast_reason: "OUT_OF_SUPPORT", demo_scenarios: "HEALTHY_RUNNING" },
        { machine_uid: "demo::idle", machine_id: "DEMO_IDLE", source_key: "local-demo", line_sel: "DEMO_LINE", latest_seen_at: "2026-01-01T00:00:00Z", operating_state: "RUNNING", operating_state_source: "SYNTHETIC_FIXTURE", admitted_to_runtime: 1, forecast_state: "AVAILABLE", demo_scenarios: "RUNNING_IDLE_RUNNING" },
        { machine_uid: "demo::unknown", machine_id: "DEMO_UNKNOWN", source_key: "local-demo", line_sel: "DEMO_LINE", latest_seen_at: "2026-01-01T00:00:00Z", operating_state: "UNKNOWN", operating_state_source: "UNAVAILABLE", admitted_to_runtime: 0, forecast_state: "PAUSED", forecast_reason: "OPERATING_STATE_UNKNOWN", demo_scenarios: "UNKNOWN_FAIL_CLOSED" },
      ] };
      return { ok: true, json: async () => body } as Response;
    }));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={client}><MemoryRouter initialEntries={["/"]}><App /></MemoryRouter></QueryClientProvider>);
    expect(await screen.findByText("LOCAL DEMO MODE")).toBeVisible();
    expect(screen.getByText("Production authorization: DISABLED")).toBeVisible();
    expect(await screen.findByText("DEMO_HEALTHY")).toBeVisible();
    expect(screen.getByText("DEMO_IDLE")).toBeVisible();
    expect(screen.getByText("DEMO_UNKNOWN")).toBeVisible();
    expect(screen.getAllByText("WITHHELD").length).toBeGreaterThan(0);
    expect(screen.getAllByText("PAUSED").length).toBeGreaterThan(0);
    expect(screen.getByText("OPERATING_STATE_UNKNOWN")).toBeVisible();
  });

  it("explains a paused forecast and runtime admission on machine detail", async () => {
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      let body: unknown;
      if (url.includes("/overview")) {
        body = { environment_mode: "DEMO", evidence_description: "Synthetic plant-shadow evidence", plant_production_authorized: false };
      } else if (url.includes("/sensors")) {
        body = { items: [] };
      } else {
        body = {
          machine: { source_key: "local-demo", line_sel: "DEMO_LINE", machine_id: "DEMO_UNKNOWN" },
          latest_prediction: { forecast_state: "PAUSED", warning_point_hours: null, critical_point_hours: null, warning_withhold_reason: "OPERATING_STATE_UNKNOWN", critical_withhold_reason: "OPERATING_STATE_UNKNOWN" },
          latest_operating_context: { operating_state: "UNKNOWN", operating_state_source: "UNAVAILABLE", operating_state_confidence: 0, admitted_to_runtime: 0, reason_code: "OPERATING_STATE_UNKNOWN", cumulative_operating_seconds: 3600 },
          latest_vibration_operating_inference: null,
          current_lifecycle: { status: "ACTIVE", left_censored: 1 },
          demo_scenarios: [{ scenario_name: "UNKNOWN_FAIL_CLOSED" }],
        };
      }
      return { ok: true, json: async () => body } as Response;
    }));
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={client}><MemoryRouter initialEntries={["/machines/demo%3A%3Aunknown"]}><App /></MemoryRouter></QueryClientProvider>);
    expect(await screen.findByText("DEMO_UNKNOWN")).toBeVisible();
    expect(screen.getByText("Runtime admitted")).toBeVisible();
    expect(screen.getByText("NO")).toBeVisible();
    expect(screen.getAllByText("OPERATING_STATE_UNKNOWN").length).toBeGreaterThan(0);
    expect(screen.getByText("PAUSED")).toBeVisible();
    expect(screen.getAllByText("Unavailable").length).toBeGreaterThan(0);
  });
});
