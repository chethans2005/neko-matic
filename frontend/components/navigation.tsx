"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export function Navigation() {
  const pathname = usePathname();

  return (
    <nav className="nav">
      <Link
        href="/training"
        className={pathname === "/training" || pathname === "/" ? "active" : ""}
      >
        ⚡ Unified Training
      </Link>
    </nav>
  );
}
