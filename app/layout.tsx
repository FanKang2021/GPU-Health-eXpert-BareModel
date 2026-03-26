import type React from "react"
import type { Metadata } from "next"
import { GeistSans } from "geist/font/sans"
import { GeistMono } from "geist/font/mono"
import "./globals.css"

export const metadata: Metadata = {
  title: "GHealthX - GPU Health eXpert",
  description: "GPU集群健康监控与诊断系统",
  generator: "v0.dev",
  icons: {
    icon: [{ url: "/logo.ico", sizes: "any" }],
    apple: [{ url: "/logo.png", sizes: "180x180", type: "image/png" }],
  },
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className={`${GeistSans.variable} ${GeistMono.variable}`}>
      <head>
        <script src="/env.js" async={false}></script>
      </head>
      <body className={GeistSans.className}>{children}</body>
    </html>
  )
}
