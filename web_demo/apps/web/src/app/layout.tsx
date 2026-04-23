import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Animal Classifier",
  description: "Masked Autoencoder animal image classifier with patch masking",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
        {children}
      </body>
    </html>
  );
}
