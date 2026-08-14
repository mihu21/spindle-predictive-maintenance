// @vitest-environment jsdom

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import "@testing-library/jest-dom/vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "./App";

afterEach(() => vi.unstubAllGlobals());

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
});
