import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import App from "./App";

describe("VGRS interface", () => {
  it("shows the workspace heading", () => {
    render(<App />);
    expect(screen.getByRole("heading", { name: /Voice Analysis Studio/i })).toBeInTheDocument();
  });
  it("discloses unavailable classification before analysis", () => {
    render(<App />);
    expect(screen.getAllByText(/Classification remains unavailable/i).length).toBeGreaterThan(0);
  });
  it("requires explicit consent before analysis", () => {
    render(<App />);
    expect(screen.getByRole("checkbox", { name: /Consent before processing/i })).not.toBeChecked();
    expect(screen.getByRole("button", { name: /Analyze recording/i })).toBeDisabled();
  });
});
