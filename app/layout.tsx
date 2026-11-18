import type React from "react"
import type { Metadata, Viewport } from "next"
import { ThemeProvider } from "@/components/theme-provider"
import ghxLogoUrl from "@/components/ghx-logo.png"
import "./globals.css"

export const metadata: Metadata = {
  title: "GHealthX - GPU Health eXpert",
  description: "GPU cluster health monitoring and diagnostics for bare-metal NVIDIA nodes",
  icons: {
    icon: [{ url: ghxLogoUrl, sizes: "32x32", type: "image/png" }],
    apple: [{ url: ghxLogoUrl, sizes: "180x180", type: "image/png" }],
  },
}

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#f5f7fb",
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
          {children}
        </ThemeProvider>
      </body>
    </html>
  )
}