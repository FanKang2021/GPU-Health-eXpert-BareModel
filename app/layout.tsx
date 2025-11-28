import type React from "react"
import type { Metadata } from "next"
import { Geist } from "next/font/google"
import { Geist_Mono } from "next/font/google"
import "./globals.css"

const geistSans = Geist({
  subsets: ["latin"],
  variable: "--font-sans",
})

const geistMono = Geist_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
})

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
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable}`}>
      <head>
        <script src="/env.js" async={false}></script>
      </head>
      <body className={geistSans.className}>{children}</body>
    </html>
  )
}
