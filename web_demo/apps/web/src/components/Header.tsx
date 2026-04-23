"use client";

import Link from "next/link";

interface HeaderProps {
  theme: "light" | "dark";
  onToggleTheme: () => void;
  onGoToUpload: () => void;
}

export default function Header({ theme, onToggleTheme, onGoToUpload }: HeaderProps) {
  return (
    <header
      style={{
        borderBottom: "1px solid var(--border)",
        background: "var(--bg-elev)",
        position: "sticky",
        top: 0,
        zIndex: 10,
      }}
    >
      <div
        style={{
          maxWidth: 1240,
          margin: "0 auto",
          padding: "14px 32px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 24,
        }}
      >
        {/* Brand */}
        <Link
          href="/"
          aria-label="Go to upload page"
          onClick={(e) => {
            e.preventDefault();
            onGoToUpload();
          }}
          style={{
            display: "flex",
            alignItems: "center",
            gap: 10,
            color: "inherit",
            textDecoration: "none",
          }}
        >
          <div
            style={{
              width: 28,
              height: 28,
              display: "grid",
              gridTemplate: "repeat(3, 1fr) / repeat(3, 1fr)",
              gap: 2,
              background: "var(--fg)",
              padding: 3,
              borderRadius: 4,
            }}
          >
            {[0, 1, 2, 3, 4, 5, 6, 7, 8].map((i) => (
              <span
                key={i}
                style={{
                  background: [1, 3, 5, 8].includes(i)
                    ? "transparent"
                    : "var(--bg-elev)",
                  borderRadius: 1,
                }}
              />
            ))}
          </div>
          <div
            style={{
              fontFamily: "var(--mono)",
              fontSize: 13,
              fontWeight: 600,
              letterSpacing: "-0.01em",
            }}
          >
            <span style={{ color: "var(--fg-muted)", fontWeight: 600 }}>
              animal-classifier
            </span>
          </div>
        </Link>

        {/* Meta */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 20,
            fontFamily: "var(--mono)",
            fontSize: 11,
            color: "var(--fg-muted)",
          }}
        >
          {[
            { k: "input", v: "224×224" },
            { k: "patch", v: "14" },
          ].map(({ k, v }) => (
            <div key={k} style={{ display: "flex", gap: 6 }}>
              <span style={{ color: "var(--fg-subtle)" }}>{k}</span>
              <span style={{ color: "var(--fg)" }}>{v}</span>
            </div>
          ))}
          <button
            onClick={onToggleTheme}
            aria-label="Toggle theme"
            style={{
              border: "1px solid var(--border)",
              background: "var(--bg-elev)",
              color: "var(--fg-muted)",
              fontFamily: "var(--mono)",
              fontSize: 11,
              padding: "5px 10px",
              borderRadius: "var(--radius)",
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              gap: 6,
            }}
          >
            <span>{theme === "dark" ? "◐" : "◑"}</span>
            <span>{theme === "dark" ? "Dark" : "Light"}</span>
          </button>
        </div>
      </div>
    </header>
  );
}
