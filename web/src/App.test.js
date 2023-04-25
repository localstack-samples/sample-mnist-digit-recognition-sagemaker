import { render, screen } from "@testing-library/react";
import React from "react";
import App from "./App";

test("renders MNIST LocalStack Demo", () => {
  render(<App />);
  const linkElement = screen.getByText(/MNIST LocalStack Demo/i);
  expect(linkElement).toBeInTheDocument();
});
