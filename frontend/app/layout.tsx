import type { Metadata } from "next";
import "./globals.css";
import { Navigation } from "@/components/navigation";

export const metadata: Metadata = {
  title: "neko-matic Dashboard",
  description: "Configurable AutoML control plane",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="orb orb-a" />
        <div className="orb orb-b" />
        <main className="shell">
          <header className="hero">
            <div>
              <h1>neko-matic</h1>
              <p>Experiment control plane for production-grade model training.</p>
            </div>
            <span className="badge">FastAPI + Next.js</span>
          </header>
          <Navigation />
          <section className="content">{children}</section>
        </main>
      </body>
    </html>
  );
}
