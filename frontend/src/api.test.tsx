import { describe, it, expect } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import App from "./App";

describe("VGRS application workspace", () => {
  it("renders the product dashboard", () => {
    render(<App />);
    expect(screen.getByRole("heading", { name: /Voice Analysis Studio/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Batch workspace/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /System settings/i })).toBeInTheDocument();
  });

  it("makes classifier availability explicit without claiming predictions", () => {
    render(<App />);
    expect(screen.getByText(/Not enabled/i)).toBeInTheDocument();
    expect(screen.getByText(/No predictions are fabricated/i)).toBeInTheDocument();
  });

  it("requires explicit consent before enabling audio analysis", () => {
    render(<App />);
    const mainNavigation = within(screen.getByRole("navigation", { name: /Main navigation/i }));
    fireEvent.click(mainNavigation.getByRole("button", { name: /Voice analysis/i }));
    expect(screen.getByRole("checkbox", { name: /I have permission to submit this recording/i })).not.toBeChecked();
    expect(screen.getByRole("button", { name: /Run acoustic analysis/i })).toBeDisabled();
  });
});
