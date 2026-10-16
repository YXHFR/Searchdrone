import { Component, type ReactNode } from "react";

interface Props {
  fallback: (reset: () => void) => ReactNode;
  children: ReactNode;
}

export class ErrorBoundary extends Component<Props, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError(): { failed: boolean } {
    return { failed: true };
  }
  render(): ReactNode {
    if (this.state.failed) return this.props.fallback(() => this.setState({ failed: false }));
    return this.props.children;
  }
}
